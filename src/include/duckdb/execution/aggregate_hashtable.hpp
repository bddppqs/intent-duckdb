//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/aggregate_hashtable.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/row_operations/row_matcher.hpp"
#include "duckdb/common/types/row/partitioned_tuple_data.hpp"
#include "duckdb/execution/base_aggregate_hashtable.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/execution/ht_entry.hpp"
#include "duckdb/storage/arena_allocator.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/types/hyperloglog.hpp"
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"

namespace duckdb {

class BlockHandle;

struct FlushMoveState;

//! Whether the aggregate sink keeps owned-dictionary string keys by reference, holding their owner; off, every sink
//! append copies the string to the heap
bool BorrowedStringKeysEnabled();

//! GroupedAggregateHashTable is a linear probing HT that is used for computing
//! aggregates
/*!
    GroupedAggregateHashTable is a HT that is used for computing aggregates. It takes
   as input the set of groups and the types of the aggregates to compute and
   stores them in the HT. It uses linear probing for collision resolution.
*/
struct AggregateHTScanState {
public:
	AggregateHTScanState() {
	}

	idx_t partition_idx = 0;
	TupleDataScanState scan_states;
};

class GroupedAggregateHashTable : public BaseAggregateHashTable {
public:
	GroupedAggregateHashTable(ClientContext &context, Allocator &allocator, vector<LogicalType> group_types,
	                          vector<LogicalType> payload_types, const vector<BoundAggregateExpression *> &aggregates,
	                          idx_t initial_capacity = InitialCapacity(), idx_t radix_bits = 0,
	                          TupleDataValidityType group_validity = TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	GroupedAggregateHashTable(ClientContext &context, Allocator &allocator, vector<LogicalType> group_types,
	                          vector<LogicalType> payload_types, vector<AggregateObject> aggregates,
	                          idx_t initial_capacity = InitialCapacity(), idx_t radix_bits = 0,
	                          TupleDataValidityType group_validity = TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	GroupedAggregateHashTable(ClientContext &context, Allocator &allocator, vector<LogicalType> group_types,
	                          TupleDataValidityType group_validity = TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	~GroupedAggregateHashTable() override;

public:
	//! The hash table load factor, when a resize is triggered
	constexpr static double LOAD_FACTOR = 1.5;

	//! Get the layout of this HT
	shared_ptr<TupleDataLayout> GetLayoutPtr();
	const TupleDataLayout &GetLayout() const;
	//! Number of groups in the HT
	idx_t Count() const;
	//! Initial capacity of the HT
	static idx_t InitialCapacity();
	//! Capacity that can hold 'count' entries without resizing
	static idx_t GetCapacityForCount(idx_t count);
	//! Current capacity of the HT
	idx_t Capacity() const;
	//! Threshold at which to resize the HT
	idx_t ResizeThreshold() const;
	static idx_t ResizeThreshold(idx_t capacity);

	//! Add the given data to the HT, computing the aggregates grouped by the
	//! data in the group chunk. When resize = true, aggregates will not be
	//! computed but instead just assigned.
	idx_t AddChunk(DataChunk &groups, DataChunk &payload, const unsafe_vector<idx_t> &filter);
	idx_t AddChunk(DataChunk &groups, Vector &group_hashes, DataChunk &payload, const unsafe_vector<idx_t> &filter);
	idx_t AddChunk(DataChunk &groups, DataChunk &payload, AggregateType filter);
	//! Add one batch of (group value, run length) pairs: one probe and one grouped run update per run instead of one
	//! per row. `groups` holds run_count group rows, `run_counts[i]` the length of run i, and `run_updates` the
	//! grouped run update of every aggregate in layout order. Returns the number of new groups
	idx_t AddRunChunk(DataChunk &groups, const uint16_t *run_counts, idx_t run_count,
	                  const vector<aggregate_grouped_run_update_t> &run_updates);
	optional_idx TryAddCompressedGroups(DataChunk &groups, DataChunk &payload, const unsafe_vector<idx_t> &filter);
	optional_idx TryAddDictionaryGroups(DataChunk &groups, DataChunk &payload, const unsafe_vector<idx_t> &filter);
	optional_idx TryAddConstantGroups(DataChunk &groups, DataChunk &payload, const unsafe_vector<idx_t> &filter);

	//! Fetch the aggregates for specific groups from the HT and place them in the result
	void FetchAggregates(DataChunk &groups, DataChunk &result);

	void InitializeScan(AggregateHTScanState &scan_state);
	bool Scan(AggregateHTScanState &scan_state, DataChunk &distinct_rows, DataChunk &payload_rows);

	//! Finds or creates groups in the hashtable using the specified group keys. The addresses vector will be filled
	//! with pointers to the groups in the hash table, and the new_groups selection vector will point to the newly
	//! created groups. The return value is the amount of newly created groups.
	idx_t FindOrCreateGroups(DataChunk &groups, Vector &group_hashes, Vector &addresses_out,
	                         SelectionVector &new_groups_out);
	idx_t FindOrCreateGroups(DataChunk &groups, Vector &addresses_out, SelectionVector &new_groups_out);
	void FindOrCreateGroups(DataChunk &groups, Vector &addresses_out);

	const PartitionedTupleData &GetPartitionedData() const;
	unique_ptr<PartitionedTupleData> AcquirePartitionedData();
	void Abandon();
	void Repartition();
	shared_ptr<ArenaAllocator> GetAggregateAllocator();

	//! Resize the HT to the specified size. Must be larger than the current size.
	void Resize(idx_t size);
	//! Resets the pointer table of the HT to all 0's
	void ClearPointerTable();
	//! Set the radix bits for this HT
	void SetRadixBits(idx_t radix_bits);
	//! Get the radix bits for this HT
	idx_t GetRadixBits() const;
	//! Get the total number of tuples sunk into this HT
	idx_t GetSinkCount() const;
	//! Get the total number of tuples materialized currently in this HT
	idx_t GetMaterializedCount() const;
	//! Skips lookups from here on out
	void SkipLookups();
	//! Enable/disable HLL
	void EnableHLL(bool enable);
	//! Whether HLL is enabled
	bool HLLEnabled() const;
	//! Get HLL count
	idx_t GetHLLUpperBound() const;

	//! Executes the filter(if any) and update the aggregates
	void Combine(GroupedAggregateHashTable &other);
	void Combine(TupleDataCollection &other_data, optional_ptr<atomic<double>> progress = nullptr);

	//! the sink's external state, read by the sink before each chunk; from the first external chunk on no key
	//! is borrowed
	void SetSinkExternal(bool external);

	//! destroys a collection's aggregate states without throwing (a destructor's caller); returns the states it
	//! could not reach (a chunk it could not pin), which are skipped
	static idx_t GlobalDictionaryDestroyStates(TupleDataCollection &data_collection, TupleDataLayout &layout,
	                                           RowOperationsState &row_state);

	//! the published dictionaries whose strings this table's rows borrow (kept alive with the rows: a later
	//! storage epoch's publish may retire a dictionary while borrowed rows still point into it)
	vector<buffer_ptr<VectorBuffer>> global_dictionary_pins;

private:
	//! The borrowed column of the next append (or INVALID_INDEX)
	idx_t GlobalDictionaryBeginAppend(DataChunk &groups);

private:
	ClientContext &context;
	//! Efficiently matches groups
	RowMatcher row_matcher;

	struct AggregateDictionaryState {
		AggregateDictionaryState();

		//! The current dictionary vector id (if any)
		string dictionary_id;
		DataChunk unique_values;
		Vector hashes;
		Vector new_dictionary_pointers;
		SelectionVector unique_entries;
		unique_ptr<Vector> dictionary_addresses;
		unsafe_unique_array<bool> found_entry;
		idx_t capacity = 0;
	};

	//! If we have this many or more radix bits, we use the unpartitioned data collection too
	static constexpr idx_t UNPARTITIONED_RADIX_BITS_THRESHOLD = 3;
	//! The number of radix bits to partition by
	idx_t radix_bits;
	//! The data of the HT
	unique_ptr<PartitionedTupleData> partitioned_data;
	unique_ptr<PartitionedTupleData> unpartitioned_data;

	//! Predicates for matching groups (always ExpressionType::COMPARE_EQUAL)
	vector<ExpressionType> predicates;

	//! The number of groups in the HT
	idx_t count;
	//! The capacity of the HT. This can be increased using GroupedAggregateHashTable::Resize
	idx_t capacity;
	//! The hash map (pointer table) of the HT: allocated data and pointer into it
	AllocatedData hash_map;
	ht_entry_t *entries;
	//! Offset of the hash column in the rows
	idx_t hash_offset;
	//! Bitmask for getting relevant bits from the hashes to determine the position
	hash_t bitmask;

	//! How many tuples went into this HT (before de-duplication)
	idx_t sink_count;
	//! If true, we just append, skipping HT lookups
	bool skip_lookups;
	//! Whether to enable HLL counting the hashes
	bool enable_hll;
	//! The associated HLL
	HyperLogLog hll;

	//! The active arena allocator used by the aggregates for their internal state
	shared_ptr<ArenaAllocator> aggregate_allocator;
	//! Owning arena allocators that this HT has data from
	vector<shared_ptr<ArenaAllocator>> stored_allocators;

	//! Append state
	struct AggregateHTAppendState {
		explicit AggregateHTAppendState(ArenaAllocator &allocator);

		PartitionedTupleDataAppendState partitioned_append_state;
		PartitionedTupleDataAppendState unpartitioned_append_state;

		Vector hashes;
		Vector ht_offsets;
		Vector hash_salts;
		SelectionVector new_groups;
		SelectionVector group_compare_vector;
		SelectionVector no_match_vector;
		Vector addresses;
		DataChunk group_chunk;
		AggregateDictionaryState dict_state;

		RowOperationsState row_state;
	} state;

private:
	//! Disabled the copy constructor
	GroupedAggregateHashTable(const GroupedAggregateHashTable &) = delete;
	//! Destroy the HT
	void Destroy();

	//! Initializes the PartitionedTupleData
	void InitializePartitionedData();
	//! Initializes the PartitionedTupleData that only has 1 partition
	void InitializeUnpartitionedData();
	//! Apply bitmask to get the entry in the HT
	inline idx_t ApplyBitMask(hash_t hash) const;
	//! Reinserts tuples (triggered by Resize)
	void ReinsertTuples(PartitionedTupleData &data);

	void UpdateAggregates(DataChunk &payload, const unsafe_vector<idx_t> &filter);

	//! Does the actual group matching / creation
	idx_t FindOrCreateGroupsInternal(DataChunk &groups, Vector &group_hashes, Vector &addresses,
	                                 SelectionVector &new_groups);

	//! the layout's single variable-size column when it is a VARCHAR group key, else INVALID_INDEX (a second
	//! variable-size column means no borrow at all: a borrowed row carries no heap)
	idx_t borrowable_key_column = DConstants::INVALID_INDEX;
	//! whether the sink has gone external
	bool sink_external = false;
	//! the borrow mask of the current group chunk - the key column when its vector is a dictionary vector whose child's
	//! string buffer owns all of its strings (with kBorrowedStringGroupKeys set and the sink not external), else
	//! INVALID_INDEX; `owner` receives that buffer
	idx_t BorrowedKeyColumn(buffer_ptr<VectorBuffer> &owner);

	//! Verify the pointer table of the HT
	void Verify();
};

} // namespace duckdb
