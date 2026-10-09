#include "duckdb/storage/table/column_segment.hpp"

#include "duckdb/common/limits.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/common/types/null_value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/struct_filter.hpp"
#include "duckdb/storage/data_pointer.hpp"
#include "duckdb/storage/table/append_state.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/planner/filter/bloom_filter.hpp"
#include "duckdb/planner/filter/selectivity_optional_filter.hpp"

#include "duckdb/common/types/uuid.hpp"
#include "duckdb/storage/compression/dict_fsst/filter_verdict_cache.hpp"
#include "duckdb/storage/object_cache.hpp"

#include <cstring>
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Create
//===--------------------------------------------------------------------===//

unique_ptr<ColumnSegment> ColumnSegment::CreatePersistentSegment(DatabaseInstance &db, BlockManager &block_manager,
                                                                 block_id_t block_id, idx_t offset,
                                                                 const LogicalType &type, idx_t count,
                                                                 CompressionType compression_type,
                                                                 BaseStatistics statistics,
                                                                 unique_ptr<ColumnSegmentState> segment_state) {
	auto &config = DBConfig::GetConfig(db);
	shared_ptr<BlockHandle> block;

	auto function = config.GetCompressionFunction(compression_type, type.InternalType());
	if (block_id != INVALID_BLOCK) {
		block = block_manager.RegisterBlock(block_id);
	}

	auto segment_size = block_manager.GetBlockSize();
	return make_uniq<ColumnSegment>(db, std::move(block), type, ColumnSegmentType::PERSISTENT, count, function,
	                                std::move(statistics), block_id, offset, segment_size, std::move(segment_state));
}

unique_ptr<ColumnSegment> ColumnSegment::CreateTransientSegment(DatabaseInstance &db,
                                                                const CompressionFunction &function,
                                                                const LogicalType &type, const idx_t segment_size,
                                                                BlockManager &block_manager) {
	// Allocate a buffer for the uncompressed segment.
	auto &buffer_manager = BufferManager::GetBufferManager(db);
	D_ASSERT(&buffer_manager == &block_manager.buffer_manager);
	auto block = buffer_manager.RegisterTransientMemory(segment_size, block_manager);

	return make_uniq<ColumnSegment>(db, std::move(block), type, ColumnSegmentType::TRANSIENT, 0U, function,
	                                BaseStatistics::CreateEmpty(type), INVALID_BLOCK, 0U, segment_size);
}

//===--------------------------------------------------------------------===//
// Construct/Destruct
//===--------------------------------------------------------------------===//
ColumnSegment::ColumnSegment(DatabaseInstance &db, shared_ptr<BlockHandle> block_p, const LogicalType &type,
                             const ColumnSegmentType segment_type, const idx_t count,
                             const CompressionFunction &function_p, BaseStatistics statistics,
                             const block_id_t block_id_p, const idx_t offset, const idx_t segment_size_p,
                             const unique_ptr<ColumnSegmentState> segment_state_p)

    : SegmentBase<ColumnSegment>(count), db(db), type(type), type_size(GetTypeIdSize(type.InternalType())),
      segment_type(segment_type), stats(std::move(statistics)), block(std::move(block_p)), function(function_p),
      block_id(block_id_p), offset(offset), segment_size(segment_size_p) {
	if (function.get().type == CompressionType::COMPRESSION_DICT_FSST) {
		dictionary_cache_key = "dict_fsst-" + UUID::ToString(UUID::GenerateRandomUUID());
	}
	if (function.get().init_segment) {
		segment_state = function.get().init_segment(*this, block_id, segment_state_p.get());
	}

	// For constant segments (CompressionType::COMPRESSION_CONSTANT) the block is a nullptr.
	D_ASSERT(!block || segment_size <= GetBlockSize());
}

ColumnSegment::ColumnSegment(ColumnSegment &other)
    : SegmentBase<ColumnSegment>(other.count.load()), db(other.db), type(std::move(other.type)),
      type_size(other.type_size), segment_type(other.segment_type), stats(std::move(other.stats)),
      block(std::move(other.block)), function(other.function), block_id(other.block_id), offset(other.offset),
      segment_size(other.segment_size), segment_state(std::move(other.segment_state)) {
	dictionary_cache_hint.store(other.dictionary_cache_hint.load(std::memory_order_relaxed), std::memory_order_relaxed);
	other.dictionary_cache_hint.store(0, std::memory_order_relaxed);
	dictionary_cache_key = std::move(other.dictionary_cache_key);
	other.dictionary_cache_key.clear();
	// For constant segments (CompressionType::COMPRESSION_CONSTANT) the block is a nullptr.
	D_ASSERT(!block || segment_size <= GetBlockSize());
}

ColumnSegment::~ColumnSegment() {
	if (!dictionary_cache_key.empty()) {
		db.GetObjectCache().Delete(dictionary_cache_key);
		dict_global::ReleaseTranslation(db, dictionary_cache_key);
		if (dict_fsst::FilterVerdictCacheEnabled()) {
			dict_fsst::DeleteFilterVerdictSlots(db.GetObjectCache(), dictionary_cache_key);
		}
	}
}

void ColumnSegment::InvalidateDictionaryCache() {
	dictionary_cache_hint.store(0, std::memory_order_relaxed);
	if (!dictionary_cache_key.empty()) {
		db.GetObjectCache().Delete(dictionary_cache_key);
		dict_global::ReleaseTranslation(db, dictionary_cache_key);
		if (dict_fsst::FilterVerdictCacheEnabled()) {
			dict_fsst::DeleteFilterVerdictSlots(db.GetObjectCache(), dictionary_cache_key);
		}
		dictionary_cache_key = "dict_fsst-" + UUID::ToString(UUID::GenerateRandomUUID());
		if (dict_fsst::SegmentCacheEnabled()) {
			dict_fsst::ResetSegmentCache(*this);
		}
	}
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
void ColumnSegment::InitializePrefetch(PrefetchState &prefetch_state, ColumnScanState &) {
	if (!block || block->BlockId() >= MAXIMUM_BLOCK) {
		// not an on-disk block
		return;
	}
	if (function.get().init_prefetch) {
		function.get().init_prefetch(*this, prefetch_state);
	} else {
		prefetch_state.AddBlock(block);
	}
}

void ColumnSegment::InitializeScan(ColumnScanState &state) {
	state.scan_state = function.get().init_scan(state.context, *this);
}

void ColumnSegment::Scan(ColumnScanState &state, idx_t scan_count, Vector &result, idx_t result_offset,
                         ScanVectorType scan_type) {
	if (scan_type == ScanVectorType::SCAN_ENTIRE_VECTOR) {
		D_ASSERT(result_offset == 0);
		Scan(state, scan_count, result);
	} else {
		D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
		ScanPartial(state, scan_count, result, result_offset);
		D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	}
}

void ColumnSegment::Select(ColumnScanState &state, idx_t scan_count, Vector &result, const SelectionVector &sel,
                           idx_t sel_count) {
	if (!function.get().select) {
		throw InternalException("ColumnSegment::Select not implemented for this compression method");
	}
	function.get().select(*this, state, scan_count, result, sel, sel_count);
}

void ColumnSegment::Filter(ColumnScanState &state, idx_t scan_count, Vector &result, SelectionVector &sel,
                           idx_t &sel_count, const TableFilter &filter, TableFilterState &filter_state) {
	if (!function.get().filter) {
		throw InternalException("ColumnSegment::Filter not implemented for this compression method");
	}
	function.get().filter(*this, state, scan_count, result, sel, sel_count, filter, filter_state);
}

FilterPropagateResult ColumnSegment::CheckDomain(ColumnScanState &state, const TableFilter &filter) {
	if (!function.get().check_domain) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	return function.get().check_domain(*this, state, filter);
}

void ColumnSegment::Skip(ColumnScanState &state) {
	function.get().skip(*this, state, state.offset_in_column - state.internal_index);
	state.internal_index = state.offset_in_column;
}

void ColumnSegment::ScanRuns(ColumnScanState &state, idx_t scan_count, RunSink &sink) {
	if (!function.get().scan_runs) {
		throw InternalException("ColumnSegment::ScanRuns not implemented for this compression method");
	}
	function.get().scan_runs(*this, state, scan_count, sink);
}

void ColumnSegment::Scan(ColumnScanState &state, idx_t scan_count, Vector &result) {
	function.get().scan_vector(*this, state, scan_count, result);
}

void ColumnSegment::ScanPartial(ColumnScanState &state, idx_t scan_count, Vector &result, idx_t result_offset) {
	function.get().scan_partial(*this, state, scan_count, result, result_offset);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void ColumnSegment::FetchRow(ColumnFetchState &state, row_t row_id, Vector &result, idx_t result_idx) {
	if (UnsafeNumericCast<idx_t>(row_id) > count) {
		throw InternalException("ColumnSegment::FetchRow - row_id out of range for segment");
	}
	function.get().fetch_row(*this, state, row_id, result, result_idx);
}

//===--------------------------------------------------------------------===//
// Append
//===--------------------------------------------------------------------===//
idx_t ColumnSegment::SegmentSize() const {
	return segment_size;
}

void ColumnSegment::Resize(idx_t new_size) {
	D_ASSERT(new_size > segment_size);
	D_ASSERT(offset == 0);
	D_ASSERT(block && new_size <= GetBlockSize());

	auto &buffer_manager = BufferManager::GetBufferManager(db);
	auto old_handle = buffer_manager.Pin(block);
	auto new_handle = buffer_manager.Allocate(MemoryTag::IN_MEMORY_TABLE, new_size);
	auto new_block = new_handle.GetBlockHandle();
	memcpy(new_handle.Ptr(), old_handle.Ptr(), segment_size);

	this->block_id = new_block->BlockId();
	this->block = std::move(new_block);
	this->segment_size = new_size;
}

void ColumnSegment::InitializeAppend(ColumnAppendState &state) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (!function.get().init_append) {
		throw InternalException("Attempting to init append to a segment without init_append method");
	}
	state.append_state = function.get().init_append(*this);
}

idx_t ColumnSegment::Append(ColumnAppendState &state, UnifiedVectorFormat &append_data, idx_t offset, idx_t count) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (!function.get().append) {
		throw InternalException("Attempting to append to a segment without append method");
	}
	return function.get().append(*state.append_state, *this, stats, append_data, offset, count);
}

idx_t ColumnSegment::FinalizeAppend(ColumnAppendState &state) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (!function.get().finalize_append) {
		throw InternalException("Attempting to call FinalizeAppend on a segment without a finalize_append method");
	}
	auto result_count = function.get().finalize_append(*this, stats);
	state.append_state.reset();
	return result_count;
}

void ColumnSegment::RevertAppend(idx_t new_count) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	if (function.get().revert_append) {
		function.get().revert_append(*this, new_count);
	}
	this->count = new_count;
}

//===--------------------------------------------------------------------===//
// Convert To Persistent
//===--------------------------------------------------------------------===//
void ColumnSegment::ConvertToPersistent(QueryContext context, optional_ptr<BlockManager> block_manager,
                                        const block_id_t block_id_p) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	InvalidateDictionaryCache();
	segment_type = ColumnSegmentType::PERSISTENT;
	block_id = block_id_p;
	offset = 0;

	if (block_id != INVALID_BLOCK) {
		D_ASSERT(!stats.statistics.IsConstant());
		// Non-constant block: write the block to disk.
		// The block data already exists in memory, so we alter the metadata,
		// which ensures that the buffer points to an on-disk block.
		block = block_manager->ConvertToPersistent(context, block_id, std::move(block));
		return;
	}

	// Constant block: no need to write anything to disk besides the stats (metadata).
	// I.e., we do not need to write an actual block.
	// Thus, we set the compression function to constant and reset the block buffer.
	D_ASSERT(stats.statistics.IsConstant());
	auto &config = DBConfig::GetConfig(db);
	function = config.GetCompressionFunction(CompressionType::COMPRESSION_CONSTANT, type.InternalType());
	block.reset();
}

void ColumnSegment::MarkAsPersistent(shared_ptr<BlockHandle> block_p, uint32_t offset_p) {
	D_ASSERT(segment_type == ColumnSegmentType::TRANSIENT);
	block_id = block_p->BlockId();
	SetBlock(std::move(block_p), offset_p);
}

void ColumnSegment::SetBlock(shared_ptr<BlockHandle> block_p, uint32_t offset_p) {
	InvalidateDictionaryCache();
	segment_type = ColumnSegmentType::PERSISTENT;
	offset = offset_p;
	block = std::move(block_p);
}

DataPointer ColumnSegment::GetDataPointer(idx_t row_start) {
	if (segment_type != ColumnSegmentType::PERSISTENT) {
		throw InternalException("Attempting to call ColumnSegment::GetDataPointer on a transient segment");
	}
	// set up the data pointer directly using the data from the persistent segment
	DataPointer pointer(stats.statistics.Copy());
	pointer.block_pointer.block_id = GetBlockId();
	pointer.block_pointer.offset = NumericCast<uint32_t>(GetBlockOffset());
	pointer.row_start = row_start;
	pointer.tuple_count = count;
	pointer.compression_type = function.get().type;
	if (function.get().serialize_state) {
		pointer.segment_state = function.get().serialize_state(*this);
	}
	return pointer;
}

//===--------------------------------------------------------------------===//
// Drop Segment
//===--------------------------------------------------------------------===//
void ColumnSegment::VisitBlockIds(BlockIdVisitor &visitor) const {
	if (block_id != INVALID_BLOCK) {
		visitor.Visit(block_id);
	}
	if (function.get().visit_block_ids) {
		function.get().visit_block_ids(*this, visitor);
	}
}

//===--------------------------------------------------------------------===//
// Filter Selection
//===--------------------------------------------------------------------===//
template <class T, class OP, bool HAS_NULL>
static idx_t TemplatedFilterSelection(UnifiedVectorFormat &vdata, T predicate, SelectionVector &sel,
                                      idx_t approved_tuple_count, SelectionVector &result_sel) {
	auto &mask = vdata.validity;
	auto vec = UnifiedVectorFormat::GetData<T>(vdata);
	idx_t result_count = 0;
	for (idx_t i = 0; i < approved_tuple_count; i++) {
		auto idx = sel.get_index(i);
		auto vector_idx = vdata.sel->get_index(idx);
		bool comparison_result =
		    (!HAS_NULL || mask.RowIsValid(vector_idx)) && OP::Operation(vec[vector_idx], predicate);
		result_sel.set_index(result_count, idx);
		result_count += comparison_result;
	}
	return result_count;
}

template <class T>
static void FilterSelectionSwitch(UnifiedVectorFormat &vdata, T predicate, SelectionVector &sel,
                                  idx_t &approved_tuple_count, ExpressionType comparison_type) {
	SelectionVector new_sel(approved_tuple_count);
	auto &mask = vdata.validity;
	// the inplace loops take the result as the last parameter
	switch (comparison_type) {
	case ExpressionType::COMPARE_EQUAL: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, Equals, false>(vdata, predicate, sel, approved_tuple_count, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, Equals, true>(vdata, predicate, sel, approved_tuple_count, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_NOTEQUAL: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, NotEquals, false>(vdata, predicate, sel, approved_tuple_count, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, NotEquals, true>(vdata, predicate, sel, approved_tuple_count, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_LESSTHAN: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, LessThan, false>(vdata, predicate, sel, approved_tuple_count, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, LessThan, true>(vdata, predicate, sel, approved_tuple_count, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_GREATERTHAN: {
		if (mask.AllValid()) {
			approved_tuple_count =
			    TemplatedFilterSelection<T, GreaterThan, false>(vdata, predicate, sel, approved_tuple_count, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, GreaterThan, true>(vdata, predicate, sel, approved_tuple_count, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_LESSTHANOREQUALTO: {
		if (mask.AllValid()) {
			approved_tuple_count = TemplatedFilterSelection<T, LessThanEquals, false>(vdata, predicate, sel,
			                                                                          approved_tuple_count, new_sel);
		} else {
			approved_tuple_count =
			    TemplatedFilterSelection<T, LessThanEquals, true>(vdata, predicate, sel, approved_tuple_count, new_sel);
		}
		break;
	}
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO: {
		if (mask.AllValid()) {
			approved_tuple_count = TemplatedFilterSelection<T, GreaterThanEquals, false>(vdata, predicate, sel,
			                                                                             approved_tuple_count, new_sel);
		} else {
			approved_tuple_count = TemplatedFilterSelection<T, GreaterThanEquals, true>(vdata, predicate, sel,
			                                                                            approved_tuple_count, new_sel);
		}
		break;
	}
	default:
		throw NotImplementedException("Unknown comparison type for filter pushed down to table!");
	}
	sel.Initialize(new_sel);
}

//===--------------------------------------------------------------------===//
// OR of equality comparisons, one pass
//===--------------------------------------------------------------------===//
// A pushed OR of k equality comparisons with constants on one integer column (an OR of equalities, or the keys a
// first-keys aggregate pushes) evaluated in one pass over the selection instead of k: the constants go into a table
// without collisions (a multiplicative hash; every empty slot holds the first constant, so a row equals its slot's
// constant iff it equals some constant), and each row costs one hash, one load and one compare whatever k is. The
// output is the per-child loop's exactly: the rows that pass, ordered by the first child they pass, in selection order
// within a child.
static constexpr idx_t OR_EQUALS_MIN_KEYS = 2;
static constexpr idx_t OR_EQUALS_MAX_KEYS = 32;
static constexpr idx_t OR_EQUALS_MAX_SLOT_BITS = 10;

template <class T>
struct OrEqualsKeyTable {
	T key[idx_t(1) << OR_EQUALS_MAX_SLOT_BITS];
	uint8_t child[idx_t(1) << OR_EQUALS_MAX_SLOT_BITS];
	uint64_t multiplier;
	idx_t shift;

	inline idx_t Slot(T value) const {
		using UNSIGNED = typename MakeUnsigned<T>::type;
		return UnsafeNumericCast<idx_t>((uint64_t(UNSIGNED(value)) * multiplier) >> shift);
	}

	//! Places keys[0..count) without collisions (a repeated key keeps its first child); false if no multiplier of the
	//! list does it at up to 2^OR_EQUALS_MAX_SLOT_BITS slots
	bool Build(const T *keys, idx_t count) {
		static constexpr uint64_t MULTIPLIERS[] = {0x9E3779B97F4A7C15ULL, 0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL,
		                                           0xD6E8FEB86659FD93ULL, 0xFF51AFD7ED558CCDULL, 0xC4CEB9FE1A85EC53ULL,
		                                           0x94D049BB133111EBULL, 0xBF58476D1CE4E5B9ULL};
		static constexpr uint8_t EMPTY = 0xFF;
		idx_t bits = 4;
		while ((idx_t(1) << bits) < 4 * count) {
			bits++;
		}
		for (; bits <= OR_EQUALS_MAX_SLOT_BITS; bits++) {
			const idx_t slots = idx_t(1) << bits;
			shift = 64 - bits;
			for (auto m : MULTIPLIERS) {
				multiplier = m;
				memset(child, EMPTY, slots);
				bool placed = true;
				for (idx_t k = 0; k < count && placed; k++) {
					const auto s = Slot(keys[k]);
					if (child[s] == EMPTY) {
						key[s] = keys[k];
						child[s] = UnsafeNumericCast<uint8_t>(k);
					} else if (key[s] != keys[k]) {
						placed = false;
					}
				}
				if (!placed) {
					continue;
				}
				for (idx_t s = 0; s < slots; s++) {
					if (child[s] == EMPTY) {
						key[s] = keys[0];
						child[s] = 0;
					}
				}
				return true;
			}
		}
		return false;
	}
};

//! The rows that pass, in selection order, with the first child each passes; returns their count
template <class T, bool HAS_NULL>
static idx_t OrEqualsSelection(UnifiedVectorFormat &vdata, const OrEqualsKeyTable<T> &table, SelectionVector &sel,
                               idx_t approved_tuple_count, SelectionVector &matched, uint8_t *first_child) {
	auto &mask = vdata.validity;
	auto vec = UnifiedVectorFormat::GetData<T>(vdata);
	idx_t match_count = 0;
	for (idx_t i = 0; i < approved_tuple_count; i++) {
		const auto idx = sel.get_index(i);
		const auto vector_idx = vdata.sel->get_index(idx);
		const T value = vec[vector_idx];
		const auto s = table.Slot(value);
		const bool hit = (!HAS_NULL || mask.RowIsValid(vector_idx)) && table.key[s] == value;
		matched.set_index(match_count, idx);
		first_child[match_count] = table.child[s];
		match_count += hit;
	}
	return match_count;
}

template <class T>
static bool OrEqualsFilterSelection(UnifiedVectorFormat &vdata, const ConjunctionOrFilter &filter, SelectionVector &sel,
                                    idx_t &approved_tuple_count) {
	const idx_t key_count = filter.child_filters.size();
	T keys[OR_EQUALS_MAX_KEYS];
	for (idx_t k = 0; k < key_count; k++) {
		keys[k] = filter.child_filters[k]->Cast<ConstantFilter>().constant.GetValueUnsafe<T>();
	}
	OrEqualsKeyTable<T> table;
	if (!table.Build(keys, key_count)) {
		return false;
	}
	SelectionVector matched(approved_tuple_count);
	auto first_child = make_unsafe_uniq_array_uninitialized<uint8_t>(MaxValue<idx_t>(approved_tuple_count, 1));
	idx_t match_count;
	if (vdata.validity.AllValid()) {
		match_count = OrEqualsSelection<T, false>(vdata, table, sel, approved_tuple_count, matched, first_child.get());
	} else {
		match_count = OrEqualsSelection<T, true>(vdata, table, sel, approved_tuple_count, matched, first_child.get());
	}
	// the per-child loop's order: grouped by the first child passed (a stable counting sort), selection order within;
	// when every row passes the same first child that is the selection order already
	bool one_child = true;
	for (idx_t i = 1; i < match_count && one_child; i++) {
		one_child = first_child[i] == first_child[0];
	}
	approved_tuple_count = match_count;
	if (one_child) {
		sel.Initialize(matched);
		return true;
	}
	idx_t offsets[OR_EQUALS_MAX_KEYS + 1] = {0};
	for (idx_t i = 0; i < match_count; i++) {
		offsets[first_child[i] + 1]++;
	}
	for (idx_t k = 0; k < key_count; k++) {
		offsets[k + 1] += offsets[k];
	}
	SelectionVector result_sel(match_count);
	for (idx_t i = 0; i < match_count; i++) {
		result_sel.set_index(offsets[first_child[i]]++, matched.get_index(i));
	}
	sel.Initialize(result_sel);
	return true;
}

//! The one-pass form of an OR of equality comparisons with constants on an integer column; false (nothing done) for any
//! other filter, type or count
static bool TryOrEqualsFilterSelection(SelectionVector &sel, Vector &vector, UnifiedVectorFormat &vdata,
                                       const ConjunctionOrFilter &filter, idx_t &approved_tuple_count) {
	const idx_t key_count = filter.child_filters.size();
	if (!kOrEqualsFilterOnePass || key_count < OR_EQUALS_MIN_KEYS || key_count > OR_EQUALS_MAX_KEYS) {
		return false;
	}
	for (auto &child : filter.child_filters) {
		if (child->filter_type != TableFilterType::CONSTANT_COMPARISON) {
			return false;
		}
		auto &constant_filter = child->Cast<ConstantFilter>();
		if (constant_filter.comparison_type != ExpressionType::COMPARE_EQUAL || constant_filter.constant.IsNull()) {
			return false;
		}
	}
	switch (vector.GetType().InternalType()) {
	case PhysicalType::INT8:
		return OrEqualsFilterSelection<int8_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::INT16:
		return OrEqualsFilterSelection<int16_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::INT32:
		return OrEqualsFilterSelection<int32_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::INT64:
		return OrEqualsFilterSelection<int64_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::UINT8:
		return OrEqualsFilterSelection<uint8_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::UINT16:
		return OrEqualsFilterSelection<uint16_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::UINT32:
		return OrEqualsFilterSelection<uint32_t>(vdata, filter, sel, approved_tuple_count);
	case PhysicalType::UINT64:
		return OrEqualsFilterSelection<uint64_t>(vdata, filter, sel, approved_tuple_count);
	default:
		return false;
	}
}

template <bool IS_NULL>
static idx_t TemplatedNullSelection(UnifiedVectorFormat &vdata, SelectionVector &sel, idx_t &approved_tuple_count) {
	auto &mask = vdata.validity;
	if (mask.AllValid()) {
		// no NULL values
		if (IS_NULL) {
			approved_tuple_count = 0;
			return 0;
		} else {
			return approved_tuple_count;
		}
	} else {
		SelectionVector result_sel(approved_tuple_count);
		idx_t result_count = 0;
		for (idx_t i = 0; i < approved_tuple_count; i++) {
			auto idx = sel.get_index(i);
			auto vector_idx = vdata.sel->get_index(idx);
			if (mask.RowIsValid(vector_idx) != IS_NULL) {
				result_sel.set_index(result_count++, idx);
			}
		}
		sel.Initialize(result_sel);
		approved_tuple_count = result_count;
		return result_count;
	}
}

idx_t ColumnSegment::FilterSelection(SelectionVector &sel, Vector &vector, UnifiedVectorFormat &vdata,
                                     const TableFilter &filter, TableFilterState &filter_state, idx_t scan_count,
                                     idx_t &approved_tuple_count) {
	switch (filter.filter_type) {
	case TableFilterType::OPTIONAL_FILTER: {
		auto &opt_filter = filter.Cast<OptionalFilter>();
		return opt_filter.FilterSelection(sel, vector, vdata, filter_state, scan_count, approved_tuple_count);
	}
	case TableFilterType::CONJUNCTION_OR: {
		auto &conjunction_or = filter.Cast<ConjunctionOrFilter>();
		if (TryOrEqualsFilterSelection(sel, vector, vdata, conjunction_or, approved_tuple_count)) {
			return approved_tuple_count;
		}
		// similar to the CONJUNCTION_AND, but we need to take care of the SelectionVectors (OR all of them)
		auto &state = filter_state.Cast<ConjunctionOrFilterState>();
		idx_t count_total = 0;
		SelectionVector result_sel(approved_tuple_count);
		for (idx_t child_idx = 0; child_idx < conjunction_or.child_filters.size(); child_idx++) {
			auto &child_filter = *conjunction_or.child_filters[child_idx];
			SelectionVector temp_sel;
			temp_sel.Initialize(sel);
			idx_t temp_tuple_count = approved_tuple_count;
			idx_t temp_count = FilterSelection(temp_sel, vector, vdata, child_filter, *state.child_states[child_idx],
			                                   scan_count, temp_tuple_count);
			// tuples passed, move them into the actual result vector
			for (idx_t i = 0; i < temp_count; i++) {
				auto new_idx = temp_sel.get_index(i);
				bool is_new_idx = true;
				for (idx_t res_idx = 0; res_idx < count_total; res_idx++) {
					if (result_sel.get_index(res_idx) == new_idx) {
						is_new_idx = false;
						break;
					}
				}
				if (is_new_idx) {
					result_sel.set_index(count_total++, new_idx);
				}
			}
		}
		sel.Initialize(result_sel);
		approved_tuple_count = count_total;
		return approved_tuple_count;
	}
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction_and = filter.Cast<ConjunctionAndFilter>();
		auto &state = filter_state.Cast<ConjunctionAndFilterState>();
		for (idx_t child_idx = 0; child_idx < conjunction_and.child_filters.size(); child_idx++) {
			auto &child_filter = *conjunction_and.child_filters[child_idx];
			FilterSelection(sel, vector, vdata, child_filter, *state.child_states[child_idx], scan_count,
			                approved_tuple_count);
		}
		return approved_tuple_count;
	}
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		switch (vector.GetType().InternalType()) {
		case PhysicalType::UINT8: {
			auto predicate = UTinyIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<uint8_t>(vdata, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type);
			break;
		}
		case PhysicalType::UINT16: {
			auto predicate = USmallIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<uint16_t>(vdata, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type);
			break;
		}
		case PhysicalType::UINT32: {
			auto predicate = UIntegerValue::Get(constant_filter.constant);
			FilterSelectionSwitch<uint32_t>(vdata, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type);
			break;
		}
		case PhysicalType::UINT64: {
			auto predicate = UBigIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<uint64_t>(vdata, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type);
			break;
		}
		case PhysicalType::INT8: {
			auto predicate = TinyIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<int8_t>(vdata, predicate, sel, approved_tuple_count, constant_filter.comparison_type);
			break;
		}
		case PhysicalType::INT16: {
			auto predicate = SmallIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<int16_t>(vdata, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type);
			break;
		}
		case PhysicalType::INT32: {
			auto predicate = IntegerValue::Get(constant_filter.constant);
			FilterSelectionSwitch<int32_t>(vdata, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type);
			break;
		}
		case PhysicalType::INT64: {
			auto predicate = BigIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<int64_t>(vdata, predicate, sel, approved_tuple_count,
			                               constant_filter.comparison_type);
			break;
		}
		case PhysicalType::INT128: {
			auto predicate = HugeIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<hugeint_t>(vdata, predicate, sel, approved_tuple_count,
			                                 constant_filter.comparison_type);
			break;
		}
		case PhysicalType::UINT128: {
			auto predicate = UhugeIntValue::Get(constant_filter.constant);
			FilterSelectionSwitch<uhugeint_t>(vdata, predicate, sel, approved_tuple_count,
			                                  constant_filter.comparison_type);
			break;
		}
		case PhysicalType::FLOAT: {
			auto predicate = FloatValue::Get(constant_filter.constant);
			FilterSelectionSwitch<float>(vdata, predicate, sel, approved_tuple_count, constant_filter.comparison_type);
			break;
		}
		case PhysicalType::DOUBLE: {
			auto predicate = DoubleValue::Get(constant_filter.constant);
			FilterSelectionSwitch<double>(vdata, predicate, sel, approved_tuple_count, constant_filter.comparison_type);
			break;
		}
		case PhysicalType::VARCHAR: {
			auto predicate = string_t(StringValue::Get(constant_filter.constant));
			FilterSelectionSwitch<string_t>(vdata, predicate, sel, approved_tuple_count,
			                                constant_filter.comparison_type);
			break;
		}
		case PhysicalType::BOOL: {
			auto predicate = BooleanValue::Get(constant_filter.constant);
			FilterSelectionSwitch<bool>(vdata, predicate, sel, approved_tuple_count, constant_filter.comparison_type);
			break;
		}
		default:
			throw InvalidTypeException(vector.GetType(), "Invalid type for filter pushed down to table comparison");
		}
		return approved_tuple_count;
	}
	case TableFilterType::DYNAMIC_FILTER: {
		// a Top-N bound applied row by row (TopN::PushdownDynamicFilters): snapshot the bound under its lock; while it is
		// not set every row passes, otherwise the snapshot is applied as the constant comparison it holds
		auto &dynamic_filter = filter.Cast<DynamicFilter>();
		if (!dynamic_filter.filter_data) {
			return approved_tuple_count;
		}
		const ConstantFilter *published;
		if (dynamic_filter.filter_data->LoadPublished(published)) {
			// the immutable copy of the bound set last, read without the lock (kTopNBoundLockFree)
			if (!published) {
				return approved_tuple_count;
			}
			return FilterSelection(sel, vector, vdata, *published, filter_state, scan_count, approved_tuple_count);
		}
		auto comparison_type = ExpressionType::INVALID;
		Value constant;
		{
			auto &filter_data = *dynamic_filter.filter_data;
			lock_guard<mutex> l(filter_data.lock);
			if (!filter_data.initialized) {
				return approved_tuple_count;
			}
			auto &bound = *filter_data.filter;
			comparison_type = bound.comparison_type;
			constant = bound.constant;
		}
		ConstantFilter snapshot(comparison_type, std::move(constant));
		return FilterSelection(sel, vector, vdata, snapshot, filter_state, scan_count, approved_tuple_count);
	}
	case TableFilterType::IS_NULL: {
		return TemplatedNullSelection<true>(vdata, sel, approved_tuple_count);
	}
	case TableFilterType::IS_NOT_NULL: {
		return TemplatedNullSelection<false>(vdata, sel, approved_tuple_count);
	}
	case TableFilterType::STRUCT_EXTRACT: {
		auto &struct_filter = filter.Cast<StructFilter>();
		// Apply the filter on the child vector
		auto &child_vec = StructVector::GetEntries(vector)[struct_filter.child_idx];
		UnifiedVectorFormat child_data;
		child_vec->ToUnifiedFormat(scan_count, child_data);
		return FilterSelection(sel, *child_vec, child_data, *struct_filter.child_filter, filter_state, scan_count,
		                       approved_tuple_count);
	}
	case TableFilterType::BLOOM_FILTER: {
		auto &bloom_filter = filter.Cast<BFTableFilter>();
		auto &state = filter_state.Cast<BFTableFilterState>();
		return bloom_filter.Filter(vector, sel, approved_tuple_count, state);
	}
	case TableFilterType::EXPRESSION_FILTER: {
		auto &state = filter_state.Cast<ExpressionFilterState>();
		SelectionVector result_sel(approved_tuple_count);
		if (scan_count > STANDARD_VECTOR_SIZE) {
			// scan count is > vector size - split up the vector into multiple chunks
			idx_t offset = 0;
			idx_t result_offset = 0;
			idx_t current_sel_offset = 0;
			SelectionVector current_sel(approved_tuple_count);
			while (offset < scan_count) {
				idx_t chunk_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, scan_count - offset);
				idx_t chunk_end = offset + chunk_count;
				DataChunk chunk;
				chunk.data.emplace_back(vector, offset, chunk_end);
				chunk.SetCardinality(chunk_count);

				// construct the relevant selection vector for the current chunk (offset ... offset + chunk_count)
				idx_t current_count = 0;
				for (; current_sel_offset < approved_tuple_count; current_sel_offset++) {
					auto sel_index = sel.get_index(current_sel_offset);
					if (sel_index >= chunk_end) {
						// exhausted the chunk
						break;
					}
					if (sel_index < offset) {
						throw InternalException("sel_index < offset in expression filter");
					}
					current_sel.set_index(current_count++, sel_index - offset);
				}
				if (current_count == 0) {
					// no matching tuples in this chunk
					offset += chunk_count;
					continue;
				}
				auto current_result_data = result_sel.data() + result_offset;
				SelectionVector current_result_sel(current_result_data);
				idx_t new_matches =
				    state.executor.SelectExpression(chunk, current_result_sel, current_sel, current_count);
				// increment all matches by the offset
				for (idx_t i = 0; i < new_matches; i++) {
					current_result_data[i] += offset;
				}
				result_offset += new_matches;
				offset += chunk_count;
			}
			approved_tuple_count = result_offset;
		} else {
			// standard case: we can handle everything at once - run the expression once
			DataChunk chunk;
			chunk.data.emplace_back(vector);
			chunk.SetCardinality(scan_count);
			approved_tuple_count = state.executor.SelectExpression(chunk, result_sel, sel, approved_tuple_count);
		}
		sel.Initialize(result_sel);
		return approved_tuple_count;
	}
	default:
		throw InternalException("FIXME: unsupported type for filter selection");
	}
}

const CompressionFunction &ColumnSegment::GetCompressionFunction() {
	return function.get();
}

} // namespace duckdb
