//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/aggregate/run_aggregate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/storage/arena_allocator.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/aggregate/ungrouped_aggregate_state.hpp"
#include "duckdb/function/compression_function.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {
class ClientContext;
class FusedAggregateGlobalState;
class FusedAggregateLocalState;
class GlobalSinkState;
class LocalSinkState;
class PhysicalHashAggregate;
class PhysicalUngroupedAggregate;
class RunAggregatePartial;

//! One scan column consumed by run-aware ungrouped aggregates
struct RunAggregateColumn {
	//! Position of the column in the table scan's column list (the scan chunk order)
	idx_t scan_column_index;
	//! The column's storage type
	LogicalType storage_type;
	//! Optional exact widening integer cast applied to the run values (its child references index 0)
	unique_ptr<Expression> cast;
	//! Indexes of the aggregates consuming this column
	vector<idx_t> aggregates;
};

//! Plan-time descriptor shared by an ungrouped aggregate and its unfiltered seq_scan: run-eligible vectors are
//! aggregated by the scan from (value, run length) pairs into per-thread partial states, combined into the shared
//! states once per row group and merged into the aggregate's global state once at Finalize.
class RunAggregateData {
public:
	static constexpr idx_t RUN_BATCH_MAX_RUNS = 2048;

	RunAggregateData(const PhysicalUngroupedAggregate &op, vector<RunAggregateColumn> columns, vector<idx_t> nullary,
	                 vector<aggregate_run_update_t> run_updates, string column_names);
	//! Grouped form: one run-eligible group column, every aggregate carrying a grouped run update
	RunAggregateData(const PhysicalHashAggregate &op, vector<RunAggregateColumn> columns,
	                 vector<aggregate_grouped_run_update_t> grouped_run_updates, string column_names);
	~RunAggregateData();

	//! Attach a descriptor to the aggregate and its scan when the plan shape is eligible (no-op otherwise)
	static void TryAttach(ClientContext &context, PhysicalUngroupedAggregate &op);
	//! The grouped counterpart: a GROUP BY over one run-eligible integer column with COUNT-family aggregates
	static void TryAttachGrouped(ClientContext &context, PhysicalHashAggregate &op);

	bool IsGrouped() const {
		return grouped_op != nullptr;
	}

	//! (Re)initialize the shared partial states for one execution
	void Reset(Allocator &client_allocator);
	//! Merge the shared states into the aggregate's global state exactly once
	void CombineInto(GlobalUngroupedAggregateState &gstate);
	unique_ptr<RunAggregatePartial> CreatePartial(ClientContext &context);

	//! Grouped: remember the radix table's global sink state of this execution and drop the previous partials; the
	//! run kind also passes the fused kernel's global state, which then takes the run batches
	void ResetGrouped(GlobalSinkState &radix_sink, optional_ptr<FusedAggregateGlobalState> fused_sink = nullptr);
	//! Grouped: create one thread-local radix sink state, owned here so it outlives the scan's local state
	LocalSinkState &CreateGroupedLocal(ClientContext &context);
	//! Grouped: sink one batch of (group value, run length) pairs into a thread-local radix sink state
	void SinkGroupedRuns(ClientContext &context, DataChunk &groups, const uint16_t *counts, idx_t run_count,
	                     LocalSinkState &lstate);
	//! Grouped: hand the thread-local sink states to the aggregate's Finalize, leaving none behind
	vector<unique_ptr<LocalSinkState>> TakeGroupedPartials();
	//! run kind: one thread-local fused state paired with the thread's radix local state (its drain's target),
	//! owned here; null on the regular run path
	optional_ptr<FusedAggregateLocalState> CreateFusedLocal(LocalSinkState &radix_local);
	//! run kind: one batch of runs into the fused kernel; false when the kernel is abandoned (the caller sinks
	//! the batch through SinkGroupedRuns)
	bool SinkFusedRuns(ClientContext &context, Vector &values, const uint16_t *counts, idx_t run_count,
	                   FusedAggregateLocalState &lstate);
	//! run kind: at the aggregate's Finalize, each thread-local fused state's lists go to the kernel (or are
	//! drained into its radix local state when the kernel is abandoned), leaving none behind
	void CombineFusedPartials(ClientContext &context);

	optional_ptr<const PhysicalUngroupedAggregate> ungrouped_op;
	optional_ptr<const PhysicalHashAggregate> grouped_op;
	const vector<RunAggregateColumn> columns;
	const vector<idx_t> nullary_aggregates;
	//! The run-aware update of every aggregate (index = aggregate index), resolved once at plan time
	const vector<aggregate_run_update_t> run_updates;
	//! The grouped run-aware update of every aggregate, in the hash table's layout order (grouped form only)
	const vector<aggregate_grouped_run_update_t> grouped_run_updates;
	const string column_names;

private:
	friend class RunAggregatePartial;
	mutex lock;
	unique_ptr<ArenaAllocator> allocator;
	unique_ptr<UngroupedAggregateState> shared;
	idx_t rows = 0;
	bool merged = false;
	//! Grouped: the radix table's global sink state of the current execution and the thread-local sink states
	optional_ptr<GlobalSinkState> grouped_sink;
	vector<unique_ptr<LocalSinkState>> grouped_partials;
	//! The run kind: the fused kernel's global state of the current execution and the thread-local fused states
	optional_ptr<FusedAggregateGlobalState> fused_sink;
	vector<unique_ptr<FusedAggregateLocalState>> fused_partials;
};

//! Per-thread run batches and partial states of one table scan
class RunAggregatePartial {
public:
	RunAggregatePartial(RunAggregateData &data, ClientContext &context);
	~RunAggregatePartial();

	RunSink &GetSink(idx_t scan_column_index);
	//! Account the rows of one run-eligible vector for the nullary aggregates (COUNT(*)); a no-op when grouped, where
	//! every run carries its own length into its own group
	void AddRows(idx_t count);
	//! Apply a run batch of one column to its aggregates
	void ApplyRuns(Vector &values, const uint16_t *counts, idx_t run_count, const vector<idx_t> &aggregates);
	//! Grouped: probe the thread-local hash table once per run and apply the grouped run updates (run kind: the
	//! batch goes to the fused kernel instead, unless the kernel is abandoned)
	void ApplyGroupedRuns(Vector &values, const uint16_t *counts, idx_t run_count);
	//! Flush the batches and combine the partial states into the shared states, then reinitialize them
	void CombineIntoShared();
	//! The scan has sunk its last run: the run kind hands this thread's fused lists over (or drains them) on
	//! this thread, so the aggregate's Finalize hands over only the partials no scan finished (a no-op otherwise)
	void FinishScan();

private:
	struct ColumnSink;
	RunAggregateData &data;
	ArenaAllocator allocator;
	unique_ptr<UngroupedAggregateState> local;
	vector<unique_ptr<ColumnSink>> sinks;
	vector<RunSink *> by_scan_column;
	Vector nullary_input;
	idx_t rows = 0;
	//! Grouped: this thread's radix sink state (owned by the shared descriptor) and the one-column group chunk
	ClientContext &context;
	optional_ptr<LocalSinkState> grouped_local;
	DataChunk group_chunk;
	//! The run kind: this thread's fused state (owned by the shared descriptor); null on the standard run path
	optional_ptr<FusedAggregateLocalState> fused_local;
};

} // namespace duckdb
