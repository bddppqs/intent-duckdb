//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/radix_partitioned_hashtable.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/execution/progress_data.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"

namespace duckdb {

class GroupedAggregateHashTable;
struct AggregatePartition;

//! Arm the borrowed-heap group-key append on this thread, and return the previous arming
//! so the caller can restore it.  While it is active, GroupedAggregateHashTable::Combine creates a
//! new group by copying the fixed-width row it was created from instead of scattering the group
//! columns into a heap of its own, so a VARCHAR group key is materialised once - by the sink that
//! first stored it - rather than a second time at the sink -> finalize boundary.  The buffer
//! handles that keep those borrowed bytes resident and at a stable address are collected in
//! "pins", which the caller must keep, together with the collection it combined, for as long as
//! the merged data.  A null "pins" disarms.  Free functions, so no class an extension can
//! construct changes signature or layout.
vector<BufferHandle> *ArmBorrowedGroupKeys(vector<BufferHandle> *pins);
//! The pin sink of the borrow active on this thread, or nullptr when the append must not borrow.
vector<BufferHandle> *BorrowedGroupKeyPins();

class RadixPartitionedHashTable {
public:
	RadixPartitionedHashTable(GroupingSet &grouping_set, const GroupedAggregateData &op,
	                          TupleDataValidityType group_validity);
	unique_ptr<GroupedAggregateHashTable> CreateHT(ClientContext &context, const idx_t capacity,
	                                               const idx_t radix_bits) const;

public:
	GroupingSet &grouping_set;
	//! The indices specified in the groups_count that do not appear in the grouping_set
	unsafe_vector<idx_t> null_groups;
	const GroupedAggregateData &op;
	vector<LogicalType> group_types;
	//! The GROUPING values that belong to this hash table
	vector<Value> grouping_values;
	//! Whether there are no NULLs in the groups
	const TupleDataValidityType group_validity;

public:
	//! Sink Interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const;
	//! The same local sink state off a bare ClientContext (the run-aware scan has no ExecutionContext of the sink)
	unique_ptr<LocalSinkState> GetLocalSinkState(ClientContext &context) const;

	void Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input, DataChunk &aggregate_input_chunk,
	          const unsafe_vector<idx_t> &filter) const;
	//! Sink one batch of (group value, run length) pairs: one probe and one grouped run update per run
	void SinkRuns(ClientContext &context, GlobalSinkState &gstate, LocalSinkState &lstate, DataChunk &groups,
	              const uint16_t *run_counts, idx_t run_count,
	              const vector<aggregate_grouped_run_update_t> &run_updates) const;
	void Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const;
	void Combine(ClientContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const;
	void Finalize(ClientContext &context, GlobalSinkState &gstate) const;

public:
	//! Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context) const;

	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, GlobalSinkState &sink,
	                         OperatorSourceInput &input) const;

	ProgressData GetProgress(ClientContext &context, GlobalSinkState &sink_p, GlobalSourceState &gstate) const;

	shared_ptr<TupleDataLayout> GetLayoutPtr() const;
	const TupleDataLayout &GetLayout() const;
	idx_t MaxThreads(GlobalSinkState &sink) const;
	static void SetMultiScan(GlobalSinkState &sink);

private:
	void SetGroupingValues();
	void PopulateGroupChunk(DataChunk &group_chunk, DataChunk &input_chunk) const;

	shared_ptr<TupleDataLayout> layout_ptr;
};

} // namespace duckdb
