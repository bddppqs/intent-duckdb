#include "duckdb/execution/aggregate_hashtable.hpp"

#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/radix_partitioning.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/types/null_value.hpp"
#include "duckdb/common/types/row/tuple_data_iterator.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/ht_entry.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/function/aggregate/min_max_string_arena.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <cstdlib>

namespace duckdb {

using ValidityBytes = TupleDataLayout::ValidityBytes;

namespace {

//! the state of one borrowed-heap merge.  "source_rows" are the partial rows of the chunk
//! FindOrCreateGroupsInternal is currently probing with; "input" is the append input built from the
//! subset of them that turned out to be new groups, with every heap size zero so the append
//! allocates no heap for them and copies none of their key bytes.
struct BorrowedGroupKeyState {
	BorrowedGroupKeyState() {
		input.heap_sizes.Reference(zero_heap_sizes);
	}

	//! STANDARD_VECTOR_SIZE zeroes: the heap every borrowed row does NOT bring with it
	Vector zero_heap_sizes = Vector(LogicalType::UBIGINT, true, true);
	TupleDataChunkState input;
	Vector *source_rows = nullptr;
};

//! Set for the whole of one GroupedAggregateHashTable::Combine that borrows, and null otherwise.
//! The append block in FindOrCreateGroupsInternal is the only reader.
thread_local BorrowedGroupKeyState *borrowed_group_key_state = nullptr;

struct BorrowedGroupKeyScope {
	explicit BorrowedGroupKeyScope(BorrowedGroupKeyState &borrow_state) {
		borrowed_group_key_state = &borrow_state;
	}
	~BorrowedGroupKeyScope() {
		borrowed_group_key_state = nullptr;
	}
};

} // namespace

// A new VARCHAR group key whose bytes live in a DICT_FSST-mode storage dictionary's own string arena is appended with
// its string_t stored verbatim and no heap copy; the collection that receives the rows holds the arena's owner until the
// rows die.
// Borrowed arenas trade memory for speed (they are held outside memory_limit); below dict_global::SMALL_MEMORY_LIMIT
// they cause swap, so a sink under a smaller memory_limit appends every key with the heap copy.

bool BorrowedStringKeysEnabled() {
	return kBorrowedStringGroupKeys;
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types, vector<LogicalType> payload_types,
                                                     const vector<BoundAggregateExpression *> &bindings,
                                                     idx_t initial_capacity, idx_t radix_bits,
                                                     TupleDataValidityType group_validity)
    : GroupedAggregateHashTable(context, allocator, std::move(group_types), std::move(payload_types),
                                AggregateObject::CreateAggregateObjects(bindings), initial_capacity, radix_bits,
                                group_validity) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context, Allocator &allocator,
                                                     vector<LogicalType> group_types,
                                                     TupleDataValidityType group_validity)
    : GroupedAggregateHashTable(context, allocator, std::move(group_types), {}, vector<AggregateObject>(),
                                InitialCapacity(), 0, group_validity) {
}

GroupedAggregateHashTable::AggregateHTAppendState::AggregateHTAppendState(ArenaAllocator &allocator)
    : hashes(LogicalType::HASH), ht_offsets(LogicalType::UBIGINT), hash_salts(LogicalType::HASH),
      new_groups(STANDARD_VECTOR_SIZE), group_compare_vector(STANDARD_VECTOR_SIZE),
      no_match_vector(STANDARD_VECTOR_SIZE), addresses(LogicalType::POINTER), row_state(allocator) {
}

GroupedAggregateHashTable::GroupedAggregateHashTable(ClientContext &context_p, Allocator &allocator,
                                                     vector<LogicalType> group_types_p,
                                                     vector<LogicalType> payload_types_p,
                                                     vector<AggregateObject> aggregate_objects_p,
                                                     idx_t initial_capacity, idx_t radix_bits,
                                                     TupleDataValidityType group_validity)
    : BaseAggregateHashTable(context_p, allocator, aggregate_objects_p, std::move(payload_types_p)), context(context_p),
      radix_bits(radix_bits), count(0), capacity(0), sink_count(0), skip_lookups(false), enable_hll(false),
      aggregate_allocator(make_shared_ptr<ArenaAllocator>(allocator)), state(*aggregate_allocator) {
	// Append hash column to the end and initialise the row layout
	group_types_p.emplace_back(LogicalType::HASH);

	auto layout = make_shared_ptr<TupleDataLayout>();
	layout->Initialize(std::move(group_types_p), std::move(aggregate_objects_p), group_validity);
	layout_ptr = std::move(layout);

	hash_offset = layout_ptr->GetOffsets()[layout_ptr->ColumnCount() - 1];

	const auto &variable_columns = layout_ptr->GetVariableColumns();
	if (variable_columns.size() == 1 && variable_columns[0] < layout_ptr->ColumnCount() - 1 &&
	    layout_ptr->GetTypes()[variable_columns[0]].id() == LogicalTypeId::VARCHAR) {
		borrowable_key_column = variable_columns[0];
	}

	// Partitioned data and pointer table
	InitializePartitionedData();
	if (radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD) {
		InitializeUnpartitionedData();
	}
	Resize(initial_capacity);

	// Predicates
	const auto expr_type =
	    layout_ptr->AllValid() ? ExpressionType::COMPARE_EQUAL : ExpressionType::COMPARE_NOT_DISTINCT_FROM;
	predicates.resize(layout_ptr->ColumnCount() - 1, expr_type);
	row_matcher.Initialize(true, *layout_ptr, predicates);
}

void GroupedAggregateHashTable::InitializePartitionedData() {
	if (!partitioned_data ||
	    RadixPartitioning::RadixBitsOfPowerOfTwo(partitioned_data->PartitionCount()) != radix_bits) {
		D_ASSERT(!partitioned_data || partitioned_data->Count() == 0);
		partitioned_data = make_uniq<RadixPartitionedTupleData>(buffer_manager, layout_ptr, MemoryTag::HASH_TABLE,
		                                                        radix_bits, layout_ptr->ColumnCount() - 1);
	} else {
		partitioned_data->Reset();
	}

	D_ASSERT(GetLayout().GetAggrWidth() == layout_ptr->GetAggrWidth());
	D_ASSERT(GetLayout().GetDataWidth() == layout_ptr->GetDataWidth());
	D_ASSERT(GetLayout().GetRowWidth() == layout_ptr->GetRowWidth());

	partitioned_data->InitializeAppendState(state.partitioned_append_state,
	                                        TupleDataPinProperties::KEEP_EVERYTHING_PINNED);
}

void GroupedAggregateHashTable::InitializeUnpartitionedData() {
	D_ASSERT(radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD);
	if (!unpartitioned_data) {
		unpartitioned_data = make_uniq<RadixPartitionedTupleData>(buffer_manager, layout_ptr, MemoryTag::HASH_TABLE,
		                                                          0ULL, layout_ptr->ColumnCount() - 1);
	} else {
		unpartitioned_data->Reset();
	}
	unpartitioned_data->InitializeAppendState(state.unpartitioned_append_state,
	                                          TupleDataPinProperties::KEEP_EVERYTHING_PINNED);
}

const PartitionedTupleData &GroupedAggregateHashTable::GetPartitionedData() const {
	return *partitioned_data;
}

unique_ptr<PartitionedTupleData> GroupedAggregateHashTable::AcquirePartitionedData() {
	if (radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD) {
		// Flush/unpin unpartitioned data and append to partitioned data
		if (unpartitioned_data) {
			unpartitioned_data->FlushAppendState(state.unpartitioned_append_state);
			unpartitioned_data->Unpin();
			unpartitioned_data->Repartition(context, *partitioned_data);
		}
		InitializeUnpartitionedData();
	}

	// Flush/unpin partitioned data
	partitioned_data->FlushAppendState(state.partitioned_append_state);
	partitioned_data->Unpin();

	// Return and re-initialize
	auto result = std::move(partitioned_data);
	InitializePartitionedData();
	return result;
}

void GroupedAggregateHashTable::Abandon() {
	if (radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD) {
		// Flush/unpin unpartitioned data and append to partitioned data
		if (unpartitioned_data) {
			unpartitioned_data->FlushAppendState(state.unpartitioned_append_state);
			unpartitioned_data->Unpin();
			unpartitioned_data->Repartition(context, *partitioned_data);
		}
		InitializeUnpartitionedData();
	}

	// Start over
	ClearPointerTable();
	count = 0;

	// Resetting the id ensures the dict state is reset properly when needed
	state.dict_state.dictionary_id = string();
}

void GroupedAggregateHashTable::Repartition() {
	auto old = AcquirePartitionedData();
	D_ASSERT(old->GetPartitions().size() != partitioned_data->GetPartitions().size());
	old->Repartition(context, *partitioned_data);
}

shared_ptr<ArenaAllocator> GroupedAggregateHashTable::GetAggregateAllocator() {
	return aggregate_allocator;
}

GroupedAggregateHashTable::~GroupedAggregateHashTable() {
	Destroy();
}

void GroupedAggregateHashTable::Destroy() {
	if (!partitioned_data || partitioned_data->Count() == 0 || !layout_ptr->HasDestructor()) {
		return;
	}

	// There are aggregates with destructors: Call the destructor for each of the aggregates
	// Currently does not happen because aggregate destructors are called while scanning in RadixPartitionedHashTable
	// LCOV_EXCL_START
	for (auto &data_collection : partitioned_data->GetPartitions()) {
		if (data_collection->Count() == 0) {
			continue;
		}
		GlobalDictionaryDestroyStates(*data_collection, *layout_ptr, state.row_state);
	}
	// LCOV_EXCL_STOP
}

idx_t GroupedAggregateHashTable::GlobalDictionaryDestroyStates(TupleDataCollection &data_collection,
                                                               TupleDataLayout &layout, RowOperationsState &row_state) {
	// runs inside destructors, so it never throws: a chunk that cannot be pinned (out of memory) ends this
	// collection's walk, and its states are skipped: their own out-of-line allocations are lost, not the process
	const auto total = data_collection.Count();
	idx_t destroyed = 0;
	try {
		TupleDataChunkIterator iterator(data_collection, TupleDataPinProperties::DESTROY_AFTER_DONE, false);
		auto &row_locations = iterator.GetChunkState().row_locations;
		do {
			const auto count = iterator.GetCurrentChunkCount();
			RowOperations::DestroyStates(row_state, layout, row_locations, count);
			destroyed += count;
		} while (iterator.Next());
	} catch (std::exception &) { // NOLINT: the skipped states are counted below
	}
	try {
		data_collection.Reset();
	} catch (std::exception &) { // NOLINT: the collection's own destructor frees what Reset could not
	}
	return total > destroyed ? total - destroyed : 0;
}

shared_ptr<TupleDataLayout> GroupedAggregateHashTable::GetLayoutPtr() {
	return partitioned_data->GetLayoutPtr();
}

const TupleDataLayout &GroupedAggregateHashTable::GetLayout() const {
	return partitioned_data->GetLayout();
}

idx_t GroupedAggregateHashTable::Count() const {
	return count;
}

idx_t GroupedAggregateHashTable::InitialCapacity() {
	return STANDARD_VECTOR_SIZE * 2ULL;
}

idx_t GroupedAggregateHashTable::GetCapacityForCount(idx_t count) {
	count = MaxValue<idx_t>(InitialCapacity(), count);
	return NextPowerOfTwo(LossyNumericCast<uint64_t>(static_cast<double>(count) * LOAD_FACTOR));
}

idx_t GroupedAggregateHashTable::Capacity() const {
	return capacity;
}

idx_t GroupedAggregateHashTable::ResizeThreshold() const {
	return ResizeThreshold(Capacity());
}

idx_t GroupedAggregateHashTable::ResizeThreshold(const idx_t capacity) {
	return LossyNumericCast<idx_t>(static_cast<double>(capacity) / LOAD_FACTOR);
}

idx_t GroupedAggregateHashTable::ApplyBitMask(hash_t hash) const {
	return hash & bitmask;
}

void GroupedAggregateHashTable::Verify() {
#ifdef DEBUG
	if (skip_lookups) {
		return;
	}
	idx_t total_count = 0;
	for (idx_t i = 0; i < capacity; i++) {
		const auto &entry = entries[i];
		if (!entry.IsOccupied()) {
			continue;
		}
		auto hash = Load<hash_t>(entry.GetPointer() + hash_offset);
		D_ASSERT(entry.GetSalt() == ht_entry_t::ExtractSalt(hash));
		total_count++;
	}
	D_ASSERT(total_count == Count());
#endif
}

void GroupedAggregateHashTable::ClearPointerTable() {
	std::fill_n(entries, capacity, ht_entry_t());
}

void GroupedAggregateHashTable::SetRadixBits(idx_t radix_bits_p) {
	radix_bits = radix_bits_p;
}

idx_t GroupedAggregateHashTable::GetRadixBits() const {
	return radix_bits;
}

idx_t GroupedAggregateHashTable::GetSinkCount() const {
	return sink_count;
}

idx_t GroupedAggregateHashTable::GetMaterializedCount() const {
	auto result = partitioned_data->Count();
	if (unpartitioned_data) {
		result += unpartitioned_data->Count();
	}
	return result;
}

void GroupedAggregateHashTable::SkipLookups() {
	skip_lookups = true;
}

void GroupedAggregateHashTable::EnableHLL(bool enable) {
	enable_hll = enable;
}

bool GroupedAggregateHashTable::HLLEnabled() const {
	return enable_hll;
}

idx_t GroupedAggregateHashTable::GetHLLUpperBound() const {
	D_ASSERT(enable_hll);
	return LossyNumericCast<idx_t>((1 + HyperLogLog::GetErrorRate()) * static_cast<double>(hll.Count()));
}

void GroupedAggregateHashTable::Resize(idx_t size) {
	D_ASSERT(size >= STANDARD_VECTOR_SIZE);
	D_ASSERT(IsPowerOfTwo(size));
	if (Count() != 0 && size < capacity) {
		throw InternalException("Cannot downsize a non-empty hash table!");
	}
	D_ASSERT(Count() == 0 || Count() == GetMaterializedCount());

	capacity = size;
	hash_map = buffer_manager.GetBufferAllocator().Allocate(capacity * sizeof(ht_entry_t));
	entries = reinterpret_cast<ht_entry_t *>(hash_map.get());
	ClearPointerTable();
	bitmask = capacity - 1;

	if (Count() != 0) {
		ReinsertTuples(*partitioned_data);
		if (radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD) {
			ReinsertTuples(*unpartitioned_data);
		}
	}

	Verify();
}

static void SaltIncrementAndWrap(idx_t &offset, const idx_t &salt, const idx_t &capacity_mask) {
	// How many of the uppermost bits of the salt to determine the linear probing increment
	static constexpr idx_t INCREMENT_BITS = 5;
	// Extract the bits and make sure it's odd so we cover the entire domain when adding modulo a power of two
	offset += (salt >> (64 - INCREMENT_BITS)) | 1;
	offset &= capacity_mask;
}

void GroupedAggregateHashTable::ReinsertTuples(PartitionedTupleData &data) {
	for (auto &data_collection : data.GetPartitions()) {
		if (data_collection->Count() == 0) {
			continue;
		}
		TupleDataChunkIterator iterator(*data_collection, TupleDataPinProperties::ALREADY_PINNED, false);
		const auto row_locations = iterator.GetRowLocations();
		do {
			for (idx_t i = 0; i < iterator.GetCurrentChunkCount(); i++) {
				const auto &row_location = row_locations[i];
				const auto hash = Load<hash_t>(row_location + hash_offset);
				const auto salt = ht_entry_t::ExtractSalt(hash);

				// Find an empty entry
				auto ht_offset = ApplyBitMask(hash);
				D_ASSERT(ht_offset == hash % capacity);
				while (entries[ht_offset].IsOccupied()) {
					SaltIncrementAndWrap(ht_offset, salt, bitmask);
				}
				auto &entry = entries[ht_offset];
				D_ASSERT(!entry.IsOccupied());
				entry.SetSalt(salt);
				entry.SetPointer(row_location);
				D_ASSERT(entry.IsOccupied());
			}
		} while (iterator.Next());
	}
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload, AggregateType filter) {
	unsafe_vector<idx_t> aggregate_filter;

	auto &aggregates = layout_ptr->GetAggregates();
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggregate = aggregates[i];
		if (aggregate.aggr_type == filter) {
			aggregate_filter.push_back(i);
		}
	}
	return AddChunk(groups, payload, aggregate_filter);
}

GroupedAggregateHashTable::AggregateDictionaryState::AggregateDictionaryState()
    : hashes(LogicalType::HASH), new_dictionary_pointers(LogicalType::POINTER), unique_entries(STANDARD_VECTOR_SIZE) {
}

optional_idx GroupedAggregateHashTable::TryAddDictionaryGroups(DataChunk &groups, DataChunk &payload,
                                                               const unsafe_vector<idx_t> &filter) {
	static constexpr idx_t MAX_DICTIONARY_SIZE_THRESHOLD = 20000;
	static constexpr idx_t DICTIONARY_THRESHOLD = 2;
	// dictionary vector - check if this is a duplicate eliminated dictionary from the storage
	auto &dict_col = groups.data[0];
	// no per-table global-code array: a published global dictionary (its id set,
	// its size above MAX_DICTIONARY_SIZE_THRESHOLD) takes the row path below with the borrow
	auto opt_dict_size = DictionaryVector::DictionarySize(dict_col);
	if (!opt_dict_size.IsValid()) {
		// dict size not known - this is not a dictionary that comes from the storage
		return optional_idx();
	}
	idx_t dict_size = opt_dict_size.GetIndex();
	auto &dictionary_id = DictionaryVector::DictionaryId(dict_col);
	if (dictionary_id.empty()) {
		// dictionary has no id, we can't cache across vectors
		// only use dictionary compression if there are fewer entries than groups
		if (dict_size * DICTIONARY_THRESHOLD >= groups.size()) {
			// dictionary is too large - use regular aggregation
			return optional_idx();
		}
	} else {
		// dictionary has an id - we can cache across vectors
		// use a much larger limit for dictionary
		if (dict_size >= MAX_DICTIONARY_SIZE_THRESHOLD) {
			// dictionary is too large - use regular aggregation
			return optional_idx();
		}
	}
	auto &dictionary_vector = DictionaryVector::Child(dict_col);
	auto &offsets = DictionaryVector::SelVector(dict_col);
	auto &dict_state = state.dict_state;
	if (dict_state.dictionary_id.empty() || dict_state.dictionary_id != dictionary_id) {
		// new dictionary - initialize the index state
		if (dict_size > dict_state.capacity) {
			dict_state.dictionary_addresses = make_uniq<Vector>(LogicalType::POINTER, dict_size);
			dict_state.found_entry = make_unsafe_uniq_array<bool>(dict_size);
			dict_state.capacity = dict_size;
		}
		memset(dict_state.found_entry.get(), 0, dict_size * sizeof(bool));
		dict_state.dictionary_id = dictionary_id;
	} else if (dict_size > dict_state.capacity) {
		throw InternalException("AggregateHT - using cached dictionary data but dictionary has changed (dictionary id "
		                        "%s - dict size %d, current capacity %d)",
		                        dict_state.dictionary_id, dict_size, dict_state.capacity);
	}

	auto &found_entry = dict_state.found_entry;
	auto &unique_entries = dict_state.unique_entries;
	idx_t unique_count = 0;
	// for each of the dictionary entries - check if we have already done a look-up into the hash table
	// if we have, we can just use the cached group pointers
	for (idx_t i = 0; i < groups.size(); i++) {
		auto dict_idx = offsets.get_index(i);
		unique_entries.set_index(unique_count, dict_idx);
		unique_count += !found_entry[dict_idx];
		found_entry[dict_idx] = true;
	}
	auto &new_dictionary_pointers = dict_state.new_dictionary_pointers;
	idx_t new_group_count = 0;
	if (unique_count > 0) {
		auto &unique_values = dict_state.unique_values;
		if (unique_values.ColumnCount() == 0) {
			unique_values.InitializeEmpty(groups.GetTypes());
		}
		// slice the dictionary
		unique_values.data[0].Slice(dictionary_vector, unique_entries, unique_count);
		unique_values.SetCardinality(unique_count);
		// now we know which entries we are going to add - hash them
		auto &hashes = dict_state.hashes;
		unique_values.Hash(hashes);

		// add the dictionary groups to the hash table
		new_group_count = FindOrCreateGroups(unique_values, hashes, new_dictionary_pointers, state.new_groups);
	}
	auto &aggregates = layout_ptr->GetAggregates();
	if (aggregates.empty()) {
		// early-out - no aggregates to update
		return new_group_count;
	}

	// set the addresses that we found for each of the unique groups in the main addresses vector
	auto new_dict_addresses = FlatVector::GetData<uintptr_t>(new_dictionary_pointers);
	// for each of the new groups, add them to the global (cached) list of addresses for the dictionary
	auto &dictionary_addresses = *dict_state.dictionary_addresses;
	auto dict_addresses = FlatVector::GetData<uintptr_t>(dictionary_addresses);
	for (idx_t i = 0; i < unique_count; i++) {
		auto dict_idx = unique_entries.get_index(i);
		dict_addresses[dict_idx] = new_dict_addresses[i] + layout_ptr->GetAggrOffset();
	}
	// now set up the addresses for the aggregates
	auto result_addresses = FlatVector::GetData<uintptr_t>(state.addresses);
	for (idx_t i = 0; i < groups.size(); i++) {
		auto dict_idx = offsets.get_index(i);
		result_addresses[i] = dict_addresses[dict_idx];
	}

	// finally process the aggregates
	UpdateAggregates(payload, filter);

	return new_group_count;
}

optional_idx GroupedAggregateHashTable::TryAddConstantGroups(DataChunk &groups, DataChunk &payload,
                                                             const unsafe_vector<idx_t> &filter) {
#ifndef DEBUG
	if (groups.size() <= 1) {
		// this only has a point if we have multiple groups
		return optional_idx();
	}
#endif
	auto &dict_state = state.dict_state;
	auto &unique_values = dict_state.unique_values;
	if (unique_values.ColumnCount() == 0) {
		unique_values.InitializeEmpty(groups.GetTypes());
	}
	// slice the dictionary
	unique_values.Reference(groups);
	unique_values.SetCardinality(1);
	unique_values.Flatten();

	auto &hashes = dict_state.hashes;
	unique_values.Hash(hashes);

	// add the single constant group to the hash table
	auto &new_dictionary_pointers = dict_state.new_dictionary_pointers;
	auto new_group_count = FindOrCreateGroups(unique_values, hashes, new_dictionary_pointers, state.new_groups);

	auto &aggregates = layout_ptr->GetAggregates();
	if (aggregates.empty()) {
		// early-out - no aggregates to update
		return new_group_count;
	}

	auto new_dict_addresses = FlatVector::GetData<uintptr_t>(new_dictionary_pointers);
	auto result_addresses = FlatVector::GetData<uintptr_t>(state.addresses);
	uintptr_t aggregate_address = new_dict_addresses[0] + layout_ptr->GetAggrOffset();
	for (idx_t i = 0; i < payload.size(); i++) {
		result_addresses[i] = aggregate_address;
	}

	// process the aggregates
	// FIXME: we can use simple_update here if the aggregates support it
	UpdateAggregates(payload, filter);

	return new_group_count;
}

optional_idx GroupedAggregateHashTable::TryAddCompressedGroups(DataChunk &groups, DataChunk &payload,
                                                               const unsafe_vector<idx_t> &filter) {
	// all groups must be compressed
	if (groups.AllConstant()) {
		return TryAddConstantGroups(groups, payload, filter);
	}
	if (groups.ColumnCount() == 1 && groups.data[0].GetVectorType() == VectorType::DICTIONARY_VECTOR &&
	    groups.data[0].GetType().InternalType() != PhysicalType::STRUCT) {
		return TryAddDictionaryGroups(groups, payload, filter);
	}
	return optional_idx();
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, DataChunk &payload, const unsafe_vector<idx_t> &filter) {
	sink_count += groups.size();

	// check if we can use an optimized path that utilizes compressed vectors
	auto result = TryAddCompressedGroups(groups, payload, filter);
	if (result.IsValid()) {
		return result.GetIndex();
	}
	// otherwise append the raw values
	groups.Hash(state.hashes);

	return AddChunk(groups, state.hashes, payload, filter);
}

idx_t GroupedAggregateHashTable::GlobalDictionaryBeginAppend(DataChunk &groups) {
	if (!dict_global::DictGlobalEnabled()) {
		return DConstants::INVALID_INDEX;
	}
	idx_t borrow_column = DConstants::INVALID_INDEX;
	auto &variable_columns = layout_ptr->GetVariableColumns();
	if (variable_columns.size() == 1 && variable_columns[0] < groups.ColumnCount() &&
	    dict_global::IsPublishedVector(groups.data[variable_columns[0]])) {
		borrow_column = variable_columns[0];
		auto child = groups.data[borrow_column].GetAuxiliary();
		bool pinned = false;
		for (auto &pin : global_dictionary_pins) {
			pinned = pinned || pin.get() == child.get();
		}
		if (!pinned) {
			global_dictionary_pins.push_back(std::move(child));
		}
	}
	return borrow_column;
}

idx_t GroupedAggregateHashTable::AddRunChunk(DataChunk &groups, const uint16_t *run_counts, idx_t run_count,
                                             const vector<aggregate_grouped_run_update_t> &run_updates) {
	if (run_count == 0) {
		return 0;
	}
#ifdef DEBUG
	D_ASSERT(groups.ColumnCount() + 1 == layout_ptr->ColumnCount());
	D_ASSERT(groups.size() == run_count);
	D_ASSERT(run_updates.size() == layout_ptr->GetAggregates().size());
	for (idx_t i = 0; i < groups.ColumnCount(); i++) {
		D_ASSERT(groups.GetTypes()[i] == layout_ptr->GetTypes()[i]);
	}
#endif
	// one probe per run; the raw group values are appended exactly as AddChunk would, without the compressed-group
	// fast paths (a run batch is a flat vector of distinct-by-construction adjacent values)
	sink_count += run_count;
	groups.Hash(state.hashes);
	const auto new_group_count = FindOrCreateGroups(groups, state.hashes, state.addresses, state.new_groups);
	VectorOperations::AddInPlace(state.addresses, NumericCast<int64_t>(layout_ptr->GetAggrOffset()), run_count);

	// one grouped run update per aggregate; the addresses vector walks the payload exactly as UpdateAggregates does
	auto &aggregates = layout_ptr->GetAggregates();
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = aggregates[i];
		AggregateInputData aggr_input_data(aggr.GetFunctionData(), state.row_state.allocator);
		run_updates[i](groups.data[0], run_counts, run_count, aggr_input_data, state.addresses);
		VectorOperations::AddInPlace(state.addresses, NumericCast<int64_t>(aggr.payload_size), run_count);
	}
	Verify();
	return new_group_count;
}

void GroupedAggregateHashTable::UpdateAggregates(DataChunk &payload, const unsafe_vector<idx_t> &filter) {
	// MIN/MAX string states of this table may take their bytes from its aggregate arena
	const MinMaxArenaScope min_max_string_arena(state.row_state.allocator,
	                                               buffer_manager.GetMaxMemory() >= dict_global::SMALL_MEMORY_LIMIT);
	// Now every cell has an entry, update the aggregates
	auto &aggregates = layout_ptr->GetAggregates();
	idx_t filter_idx = 0;
	idx_t payload_idx = 0;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = aggregates[i];
		if (filter_idx >= filter.size() || i < filter[filter_idx]) {
			// Skip all the aggregates that are not in the filter
			payload_idx += aggr.child_count;
			VectorOperations::AddInPlace(state.addresses, NumericCast<int64_t>(aggr.payload_size), payload.size());
			continue;
		}
		D_ASSERT(i == filter[filter_idx]);

		if (aggr.aggr_type != AggregateType::DISTINCT && aggr.filter) {
			RowOperations::UpdateFilteredStates(state.row_state, filter_set.GetFilterData(i), aggr, state.addresses,
			                                    payload, payload_idx);
		} else {
			RowOperations::UpdateStates(state.row_state, aggr, state.addresses, payload, payload_idx, payload.size());
		}

		// Move to the next aggregate
		payload_idx += aggr.child_count;
		VectorOperations::AddInPlace(state.addresses, NumericCast<int64_t>(aggr.payload_size), payload.size());
		filter_idx++;
	}

	Verify();
}

idx_t GroupedAggregateHashTable::AddChunk(DataChunk &groups, Vector &group_hashes, DataChunk &payload,
                                          const unsafe_vector<idx_t> &filter) {
	if (groups.size() == 0) {
		return 0;
	}

#ifdef DEBUG
	D_ASSERT(groups.ColumnCount() + 1 == layout_ptr->ColumnCount());
	for (idx_t i = 0; i < groups.ColumnCount(); i++) {
		D_ASSERT(groups.GetTypes()[i] == layout_ptr->GetTypes()[i]);
	}
#endif

	const auto new_group_count = FindOrCreateGroups(groups, group_hashes, state.addresses, state.new_groups);
	VectorOperations::AddInPlace(state.addresses, NumericCast<int64_t>(layout_ptr->GetAggrOffset()), payload.size());

	UpdateAggregates(payload, filter);

	return new_group_count;
}

void GroupedAggregateHashTable::FetchAggregates(DataChunk &groups, DataChunk &result) {
#ifdef DEBUG
	groups.Verify();
	D_ASSERT(groups.ColumnCount() + 1 == layout_ptr->ColumnCount());
	for (idx_t i = 0; i < result.ColumnCount(); i++) {
		D_ASSERT(result.data[i].GetType() == payload_types[i]);
	}
#endif

	result.SetCardinality(groups);
	if (groups.size() == 0) {
		return;
	}

	// find the groups associated with the addresses
	// FIXME: this should not use the FindOrCreateGroups, creating them is unnecessary
	FindOrCreateGroups(groups, state.addresses);
	// now fetch the aggregates
	RowOperations::FinalizeStates(state.row_state, *layout_ptr, state.addresses, result, 0);
}

template <bool HAS_SEL>
static void GroupedAggregateHashTableInnerLoop(ht_entry_t *const entries, const idx_t capacity, const hash_t bitmask,
                                               const hash_t *const hash_salts, uint64_t *const ht_offsets,
                                               const SelectionVector *const sel_vector, const idx_t remaining_entries,
                                               SelectionVector &empty_vector, SelectionVector &compare_vector,
                                               idx_t &empty_count, idx_t &compare_count) {
	// For each remaining entry, figure out whether or not it belongs to a full or empty group
	for (idx_t i = 0; i < remaining_entries; i++) {
		const auto index = HAS_SEL ? sel_vector->get_index_unsafe(i) : i;
		const auto salt = hash_salts[index];
		auto &ht_offset = ht_offsets[index];

		idx_t inner_iteration_count;
		for (inner_iteration_count = 0; inner_iteration_count < capacity; inner_iteration_count++) {
			auto &entry = entries[ht_offset];
			if (!entry.IsOccupied()) { // Unoccupied: claim it
				entry.SetSalt(salt);
				empty_vector.set_index(empty_count++, index);
				break;
			}

			if (DUCKDB_LIKELY(entry.GetSalt() == salt)) { // Matching salt: compare groups
				compare_vector.set_index(compare_count++, index);
				break;
			}

			// Linear probing
			SaltIncrementAndWrap(ht_offset, salt, bitmask);
		}
		if (DUCKDB_UNLIKELY(inner_iteration_count == capacity)) {
			throw InternalException("Maximum inner iteration count reached in GroupedAggregateHashTable");
		}
	}
}

idx_t GroupedAggregateHashTable::FindOrCreateGroupsInternal(DataChunk &groups, Vector &group_hashes_v,
                                                            Vector &addresses_v, SelectionVector &new_groups_out) {
	D_ASSERT(groups.ColumnCount() + 1 == layout_ptr->ColumnCount());
	D_ASSERT(group_hashes_v.GetType() == LogicalType::HASH);
	D_ASSERT(state.ht_offsets.GetVectorType() == VectorType::FLAT_VECTOR);
	D_ASSERT(state.ht_offsets.GetType() == LogicalType::UBIGINT);
	D_ASSERT(addresses_v.GetType() == LogicalType::POINTER);
	D_ASSERT(state.hash_salts.GetType() == LogicalType::HASH);

	// Need to fit the entire vector, and resize at threshold
	const auto chunk_size = groups.size();
	if (Count() + chunk_size > capacity || Count() + chunk_size > ResizeThreshold()) {
		Verify();
		Resize(capacity * 2);
	}
	D_ASSERT(capacity - Count() >= chunk_size); // we need to be able to fit at least one vector of data

	// we start out with all entries [0, 1, 2, ..., chunk_size]
	const SelectionVector *sel_vector = FlatVector::IncrementalSelectionVector();

	// Make a chunk that references the groups and the hashes and convert to unified format
	if (state.group_chunk.ColumnCount() == 0) {
		state.group_chunk.InitializeEmpty(layout_ptr->GetTypes());
	}
	D_ASSERT(state.group_chunk.ColumnCount() == layout_ptr->GetTypes().size());
	for (idx_t grp_idx = 0; grp_idx < groups.ColumnCount(); grp_idx++) {
		state.group_chunk.data[grp_idx].Reference(groups.data[grp_idx]);
	}
	state.group_chunk.data[groups.ColumnCount()].Reference(group_hashes_v);
	state.group_chunk.SetCardinality(groups);

	// convert all vectors to unified format
	TupleDataCollection::ToUnifiedFormat(state.partitioned_append_state.chunk_state, state.group_chunk);

	// the borrowed key column and its owner, decided once per chunk from the group chunk (never from a dictionary id)
	buffer_ptr<VectorBuffer> borrowed_owner;
	const auto borrowed_column = BorrowedKeyColumn(borrowed_owner);

	group_hashes_v.Flatten(chunk_size);
	const auto hashes = FlatVector::GetData<hash_t>(group_hashes_v);

	addresses_v.Flatten(chunk_size);
	const auto addresses = FlatVector::GetData<data_ptr_t>(addresses_v);

	if (skip_lookups) {
		// Just appending now
		auto &append_chunk_state = state.partitioned_append_state.chunk_state;
		append_chunk_state.borrowed_key_column = borrowed_column;
		append_chunk_state.borrowed_owner = borrowed_owner;
		auto &thread_borrow = dict_global::ThreadBorrowColumn();
		thread_borrow = GlobalDictionaryBeginAppend(groups);
		try {
			partitioned_data->AppendUnified(state.partitioned_append_state, state.group_chunk,
			                                *FlatVector::IncrementalSelectionVector(), chunk_size);
		} catch (...) {
			thread_borrow = DConstants::INVALID_INDEX;
			throw;
		}
		thread_borrow = DConstants::INVALID_INDEX;
		append_chunk_state.borrowed_key_column = DConstants::INVALID_INDEX;
		append_chunk_state.borrowed_owner.reset();
		RowOperations::InitializeStates(*layout_ptr, state.partitioned_append_state.chunk_state.row_locations,
		                                *FlatVector::IncrementalSelectionVector(), chunk_size);

		const auto row_locations =
		    FlatVector::GetData<data_ptr_t>(state.partitioned_append_state.chunk_state.row_locations);
		const auto &row_sel = state.partitioned_append_state.reverse_partition_sel;
		for (idx_t i = 0; i < chunk_size; i++) {
			const auto row_idx = row_sel.get_index_unsafe(i);
			const auto &row_location = row_locations[row_idx];
			addresses[i] = row_location;
		}
		count += chunk_size;
		return chunk_size;
	}

	// Compute the entry in the table based on the hash using a modulo,
	// and precompute the hash salts for faster comparison below
	const auto ht_offsets = FlatVector::GetData<uint64_t>(state.ht_offsets);
	const auto hash_salts = FlatVector::GetData<hash_t>(state.hash_salts);

	// We also compute the occupied count, which is essentially useless.
	// However, this loop is branchless, while the main lookup loop below is not.
	// So, by doing the lookups here, we better amortize cache misses.
	idx_t occupied_count = 0;
	for (idx_t r = 0; r < chunk_size; r++) {
		const auto &hash = hashes[r];
		auto &ht_offset = ht_offsets[r];
		ht_offset = ApplyBitMask(hash);
		occupied_count += entries[ht_offset].IsOccupied(); // Lookup
		D_ASSERT(ht_offset == hash % capacity);
		hash_salts[r] = ht_entry_t::ExtractSalt(hash);
	}

	SelectionVector empty_vector;
	idx_t new_group_count = 0;
	idx_t remaining_entries = chunk_size;
	idx_t iteration_count;
	for (iteration_count = 0; remaining_entries > 0 && iteration_count < capacity; iteration_count++) {
		idx_t new_entry_count = 0;
		idx_t need_compare_count = 0;
		idx_t no_match_count = 0;

		// "new_groups_out" contains ALL groups for the chunk, "empty_vector" only the groups for this iteration,
		// so it's just the same selection vector, but offset by the current "new_group_count".
		empty_vector.Initialize(new_groups_out.data() + new_group_count);
		if (sel_vector->IsSet()) {
			GroupedAggregateHashTableInnerLoop<true>(entries, capacity, bitmask, hash_salts, ht_offsets, sel_vector,
			                                         remaining_entries, empty_vector, state.group_compare_vector,
			                                         new_entry_count, need_compare_count);
		} else {
			GroupedAggregateHashTableInnerLoop<false>(entries, capacity, bitmask, hash_salts, ht_offsets, sel_vector,
			                                          remaining_entries, empty_vector, state.group_compare_vector,
			                                          new_entry_count, need_compare_count);
		}
		new_group_count += new_entry_count;

		if (DUCKDB_UNLIKELY(occupied_count > new_entry_count + need_compare_count)) {
			// We use the useless occupied_count we summed above here so the variable is used,
			// and the compiler cannot optimize away the vectorized lookups above. This should never be triggered.
			throw InternalException("Internal validation failed in GroupedAggregateHashTable");
		}
		occupied_count = 0; // Have to set to 0 for next iterations

		if (new_entry_count != 0) {
			// Append everything that belongs to an empty group
			optional_ptr<PartitionedTupleData> data;
			optional_ptr<PartitionedTupleDataAppendState> append_state;
			// a borrowed merge appends the partial row the new group was created from
			// instead of scattering the group columns into a heap of our own, so the group key is
			// materialised once (by the sink) rather than twice
			const auto borrow = borrowed_group_key_state;
			if (borrow) {
				data = partitioned_data.get();
				append_state = &state.partitioned_append_state;
			} else if (radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD &&
			           new_entry_count / RadixPartitioning::NumberOfPartitions(radix_bits) <= 4) {
				TupleDataCollection::ToUnifiedFormat(state.unpartitioned_append_state.chunk_state, state.group_chunk);
				data = unpartitioned_data.get();
				append_state = &state.unpartitioned_append_state;
			} else {
				data = partitioned_data.get();
				append_state = &state.partitioned_append_state;
			}
			if (borrow) {
				const auto source_rows = FlatVector::GetData<data_ptr_t>(*borrow->source_rows);
				const auto borrowed_rows = FlatVector::GetData<data_ptr_t>(borrow->input.row_locations);
				for (idx_t new_entry_idx = 0; new_entry_idx < new_entry_count; new_entry_idx++) {
					borrowed_rows[new_entry_idx] = source_rows[empty_vector.get_index_unsafe(new_entry_idx)];
				}
				// every heap size in the input is zero, so this copies the fixed-width rows and no
				// heap: the group keys keep pointing into the collection we are combining
				data->Append(*append_state, borrow->input, new_entry_count);
				const auto appended = FlatVector::GetData<data_ptr_t>(append_state->chunk_state.row_locations);
				const auto heap_size_offset = layout_ptr->GetHeapSizeOffset();
				for (idx_t new_entry_idx = 0; new_entry_idx < new_entry_count; new_entry_idx++) {
					// the copied row owns none of the heap bytes its source row owns
					Store<idx_t>(0, appended[new_entry_idx] + heap_size_offset);
				}
			} else {
				auto &append_chunk_state = append_state->chunk_state;
				append_chunk_state.borrowed_key_column = borrowed_column;
				append_chunk_state.borrowed_owner = borrowed_owner;
				// a group column over a published global dictionary is stored verbatim (a borrow: no heap,
				// no copy) when it is the layout's only variable-size column
				auto &thread_borrow = dict_global::ThreadBorrowColumn();
				thread_borrow = GlobalDictionaryBeginAppend(groups);
				try {
					data->AppendUnified(*append_state, state.group_chunk, empty_vector, new_entry_count);
				} catch (...) {
					thread_borrow = DConstants::INVALID_INDEX;
					throw;
				}
				thread_borrow = DConstants::INVALID_INDEX;
				append_chunk_state.borrowed_key_column = DConstants::INVALID_INDEX;
				append_chunk_state.borrowed_owner.reset();
			}
			RowOperations::InitializeStates(*layout_ptr, append_state->chunk_state.row_locations,
			                                *FlatVector::IncrementalSelectionVector(), new_entry_count);

			// Set the entry pointers in the 1st part of the HT now that the data has been appended
			const auto row_locations = FlatVector::GetData<data_ptr_t>(append_state->chunk_state.row_locations);
			const auto &row_sel = append_state->reverse_partition_sel;
			for (idx_t new_entry_idx = 0; new_entry_idx < new_entry_count; new_entry_idx++) {
				const auto index = empty_vector.get_index_unsafe(new_entry_idx);
				// a borrowed append wrote the rows in new-entry order, not in chunk order
				const auto row_idx = borrow ? new_entry_idx : row_sel.get_index_unsafe(index);
				const auto &row_location = row_locations[row_idx];

				auto &entry = entries[ht_offsets[index]];

				entry.SetPointer(row_location);
				addresses[index] = row_location;
			}
		}

		if (need_compare_count != 0) {
			// Get the pointers to the rows that need to be compared
			for (idx_t need_compare_idx = 0; need_compare_idx < need_compare_count; need_compare_idx++) {
				const auto index = state.group_compare_vector.get_index_unsafe(need_compare_idx);
				const auto &entry = entries[ht_offsets[index]];
				addresses[index] = entry.GetPointer();
			}

			// Perform group comparisons
			row_matcher.Match(state.group_chunk, state.partitioned_append_state.chunk_state.vector_data,
			                  state.group_compare_vector, need_compare_count, addresses_v, &state.no_match_vector,
			                  no_match_count);
		}

		// Linear probing: each of the entries that do not match move to the next entry in the HT
		for (idx_t i = 0; i < no_match_count; i++) {
			const auto index = state.no_match_vector.get_index_unsafe(i);
			auto &ht_offset = ht_offsets[index];
			const auto salt = hash_salts[index];
			SaltIncrementAndWrap(ht_offset, salt, bitmask);
		}
		sel_vector = &state.no_match_vector;
		remaining_entries = no_match_count;
	}
	if (iteration_count == capacity) {
		throw InternalException("Maximum outer iteration count reached in GroupedAggregateHashTable");
	}

	// HLL insertion is idempotent for repeated hashes. Feed only the groups that
	// this probe actually created; this preserves the estimator state while
	// avoiding duplicate work in the enabled adaptation window.
	if (enable_hll && new_group_count != 0) {
		hll.Update(group_hashes_v, new_groups_out, new_group_count, chunk_size);
	}

	count += new_group_count;
	return new_group_count;
}

void GroupedAggregateHashTable::SetSinkExternal(const bool external) {
	sink_external = external;
}

idx_t GroupedAggregateHashTable::BorrowedKeyColumn(buffer_ptr<VectorBuffer> &owner) {
	if (borrowable_key_column == DConstants::INVALID_INDEX || sink_external || !BorrowedStringKeysEnabled() ||
	    buffer_manager.GetMaxMemory() < dict_global::SMALL_MEMORY_LIMIT) {
		return DConstants::INVALID_INDEX;
	}
	auto &key = state.group_chunk.data[borrowable_key_column];
	if (key.GetVectorType() != VectorType::DICTIONARY_VECTOR) {
		return DConstants::INVALID_INDEX;
	}
	auto &child = DictionaryVector::Child(key);
	if (child.GetVectorType() != VectorType::FLAT_VECTOR) {
		return DConstants::INVALID_INDEX;
	}
	auto auxiliary = child.GetAuxiliary();
	if (!auxiliary || auxiliary->GetBufferType() != VectorBufferType::STRING_BUFFER ||
	    !auxiliary->Cast<VectorStringBuffer>().owns_all_strings) {
		return DConstants::INVALID_INDEX;
	}
	owner = std::move(auxiliary);
	return borrowable_key_column;
}

// this is to support distinct aggregations where we need to record whether we
// have already seen a value for a group
idx_t GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &group_hashes, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	return FindOrCreateGroupsInternal(groups, group_hashes, addresses_out, new_groups_out);
}

void GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &addresses) {
	// create a dummy new_groups sel vector
	FindOrCreateGroups(groups, addresses, state.new_groups);
}

idx_t GroupedAggregateHashTable::FindOrCreateGroups(DataChunk &groups, Vector &addresses_out,
                                                    SelectionVector &new_groups_out) {
	groups.Hash(state.hashes);
	return FindOrCreateGroups(groups, state.hashes, addresses_out, new_groups_out);
}

struct FlushMoveState {
	explicit FlushMoveState(TupleDataCollection &collection_p, vector<BufferHandle> *borrowed_pins_p)
	    : collection(collection_p), borrowed_pins(borrowed_pins_p), hashes(LogicalType::HASH),
	      group_addresses(LogicalType::POINTER), new_groups_sel(STANDARD_VECTOR_SIZE) {
		const auto &layout = collection.GetLayout();
		vector<column_t> column_ids;
		column_ids.reserve(layout.ColumnCount() - 1);
		for (idx_t col_idx = 0; col_idx < layout.ColumnCount() - 1; col_idx++) {
			column_ids.emplace_back(col_idx);
		}
		// a borrowed merge leaves the group keys in these blocks, so nothing may destroy
		// them as they are scanned
		collection.InitializeScan(scan_state, column_ids,
		                          borrowed_pins ? TupleDataPinProperties::UNPIN_AFTER_DONE
		                                        : TupleDataPinProperties::DESTROY_AFTER_DONE);
		collection.InitializeScanChunk(scan_state, groups);
		hash_col_idx = layout.ColumnCount() - 1;
	}

	bool Scan() {
		if (borrowed_pins) {
			// take over the handles of the chunk we are done with, so its blocks stay
			// resident at the address the group keys we borrowed from it point into.  Taken before
			// the next chunk is fetched, exactly as BlockIteratorState does it, because the rows of
			// the chunk just processed are read until then.
			scan_state.pin_state.row_handles.acquire_handles(*borrowed_pins);
			scan_state.pin_state.heap_handles.acquire_handles(*borrowed_pins);
		}
		if (collection.Scan(scan_state, groups)) {
			collection.Gather(scan_state.chunk_state.row_locations, *FlatVector::IncrementalSelectionVector(),
			                  groups.size(), hash_col_idx, hashes, *FlatVector::IncrementalSelectionVector(), nullptr);
			return true;
		}

		collection.FinalizePinState(scan_state.pin_state);
		return false;
	}

	TupleDataCollection &collection;
	//! Where the pins of the scanned blocks go, or null when this merge does not borrow
	vector<BufferHandle> *const borrowed_pins;
	TupleDataScanState scan_state;
	DataChunk groups;

	idx_t hash_col_idx;
	Vector hashes;

	Vector group_addresses;
	SelectionVector new_groups_sel;
};

void GroupedAggregateHashTable::Combine(GroupedAggregateHashTable &other) {
	auto other_partitioned_data = other.AcquirePartitionedData();
	auto other_data = other_partitioned_data->GetUnpartitioned();
	const auto previous = ArmBorrowedGroupKeys(nullptr);
	Combine(*other_data);
	ArmBorrowedGroupKeys(previous);

	// Inherit ownership to all stored aggregate allocators
	stored_allocators.emplace_back(other.aggregate_allocator);
	for (const auto &stored_allocator : other.stored_allocators) {
		stored_allocators.emplace_back(stored_allocator);
	}
}

void GroupedAggregateHashTable::Combine(TupleDataCollection &other_data, optional_ptr<atomic<double>> progress) {
	D_ASSERT(other_data.GetLayout().GetAggrWidth() == layout_ptr->GetAggrWidth());
	D_ASSERT(other_data.GetLayout().GetDataWidth() == layout_ptr->GetDataWidth());
	D_ASSERT(other_data.GetLayout().GetRowWidth() == layout_ptr->GetRowWidth());

	if (other_data.Count() == 0) {
		return;
	}

	// borrow the group keys of the rows we are combining rather than materialising them a
	// second time.  Only for the single-partition finalize merge that arms it, and only when the
	// rows actually carry a heap and the two layouts are identical, so a row copied out of
	// "other_data" is a valid row here.
	const auto borrowed_pins = (radix_bits == 0 && !layout_ptr->AllConstant() &&
	                            other_data.GetLayout().GetTypes() == layout_ptr->GetTypes() &&
	                            other_data.GetLayout().GetRowWidth() == layout_ptr->GetRowWidth())
	                               ? BorrowedGroupKeyPins()
	                               : nullptr;

	FlushMoveState fm_state(other_data, borrowed_pins);
	BorrowedGroupKeyState borrow_state;
	unique_ptr<BorrowedGroupKeyScope> borrow_scope;
	if (borrowed_pins) {
		borrow_scope = make_uniq<BorrowedGroupKeyScope>(borrow_state);
	}

	// MIN/MAX string states combined into this table may take their bytes from its aggregate arena
	const MinMaxArenaScope min_max_string_arena(state.row_state.allocator,
	                                               buffer_manager.GetMaxMemory() >= dict_global::SMALL_MEMORY_LIMIT);
	idx_t chunk_idx = 0;
	const auto chunk_count = other_data.ChunkCount();
	while (fm_state.Scan()) {
		// Check for interrupts with each chunk
		if (context.interrupted.load(std::memory_order_relaxed)) {
			throw InterruptException();
		}
		const auto input_chunk_size = fm_state.groups.size();
		borrow_state.source_rows = &fm_state.scan_state.chunk_state.row_locations;
		FindOrCreateGroups(fm_state.groups, fm_state.hashes, fm_state.group_addresses, fm_state.new_groups_sel);
		RowOperations::CombineStates(state.row_state, *layout_ptr, fm_state.scan_state.chunk_state.row_locations,
		                             fm_state.group_addresses, input_chunk_size);
		if (layout_ptr->HasDestructor()) {
			RowOperations::DestroyStates(state.row_state, *layout_ptr, fm_state.scan_state.chunk_state.row_locations,
			                             input_chunk_size);
		}

		if (progress) {
			*progress = static_cast<double>(++chunk_idx) / static_cast<double>(chunk_count);
		}
	}

	Verify();
}

void GroupedAggregateHashTable::InitializeScan(AggregateHTScanState &scan_state) {
	scan_state.partition_idx = 0;
	vector<idx_t> group_indexes(layout_ptr->ColumnCount() - 1);
	for (idx_t i = 0; i < group_indexes.size(); i++) {
		group_indexes[i] = i;
	}

	auto &partition = partitioned_data->GetPartitions()[scan_state.partition_idx];
	partition->InitializeScan(scan_state.scan_states, group_indexes);
}

bool GroupedAggregateHashTable::Scan(AggregateHTScanState &scan_state, DataChunk &distinct_rows,
                                     DataChunk &payload_rows) {
	if (scan_state.partition_idx >= partitioned_data->PartitionCount()) {
		return false;
	}

	payload_rows.Reset();
	distinct_rows.Reset();
	auto &current_partition = partitioned_data->GetPartitions()[scan_state.partition_idx];

	if (current_partition->Scan(scan_state.scan_states, distinct_rows)) {
		FetchAggregates(distinct_rows, payload_rows);
		return true;
	} else {
		if (++(scan_state.partition_idx) >= partitioned_data->PartitionCount()) {
			return false;
		} else {
			auto &new_partition = partitioned_data->GetPartitions()[scan_state.partition_idx];
			new_partition->InitializeScan(scan_state.scan_states);
			return true;
		}
	}
}
} // namespace duckdb
