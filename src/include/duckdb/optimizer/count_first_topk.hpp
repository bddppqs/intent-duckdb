//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/count_first_topk.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
class LogicalAggregate;
class LogicalGet;
class Optimizer;

//! CountFirstTopK rewrites `TOP_N(ORDER BY count DESC LIMIT k) -> PROJECTION* -> AGGREGATE(G; count, A...) -> input`,
//! where the input is a projection/filter chain over one base-table scan, into the same Top-N, projections and
//! aggregate over a SEMI join of the input against the top-k keys of a count-only aggregate over a copy of the input
//! (inner AGGREGATE(G; count) under an inner TOP_N(count DESC LIMIT k)). The inner Top-N makes the only tie choice;
//! the unchanged aggregate computes every value from the complete groups of those keys, so every aggregate is exact.
//! Engaged only on an unfiltered scan of at least MIN_ROWS rows in which one group key is near-unique by its own
//! distinct-count statistic, with a wide payload state; every other plan is left unchanged.
class CountFirstTopK {
public:
	//! The largest admitted LIMIT (FirstKeysAggregate::MAX_KEYS)
	static constexpr const idx_t MAX_LIMIT = 1024;
	//! The smallest admitted input row count, 2^20: below it the aggregate is small and the second scan does not pay
	static constexpr const idx_t MIN_ROWS = idx_t(kCountFirstMinRows);
	//! The smallest admitted sum of the payload aggregates' state sizes
	static constexpr const idx_t MIN_PAYLOAD_STATE_BYTES = 16;
	//! The largest admitted input rows per distinct value of the near-unique group key
	static constexpr const idx_t MAX_ROWS_PER_KEY_VALUE = idx_t(kCountFirstMaxRowsPerKeyValue);

public:
	explicit CountFirstTopK(Optimizer &optimizer);

	//! Whether the rewrite is enabled (a compile-time constant)
	static bool Enabled();

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	bool TryRewrite(LogicalOperator &topn_op);
	static bool AdmitAggregate(const LogicalAggregate &aggr);
	static bool AdmitGet(const LogicalGet &get);
	//! Copy the projection/filter chain and its scan with fresh table indices; fills old -> new table index map
	unique_ptr<LogicalOperator> CopyInput(const LogicalOperator &input, unordered_map<idx_t, idx_t> &index_map);
	static void ReplaceTableIndices(Expression &expr, const unordered_map<idx_t, idx_t> &index_map);
	static void ReplaceTableIndices(LogicalOperator &op, const unordered_map<idx_t, idx_t> &index_map);
	//! Resolve a group column reference through the chain down to the scan's column index (index into column_ids)
	static bool ResolveScanColumn(const LogicalOperator &input, ColumnBinding binding, const LogicalGet &get,
	                              idx_t &column_index);
	//! Keep in the copied chain only what `needed` (bindings at its top) reads; returns the scan's binding remap
	static void PruneCopy(LogicalOperator &op, const column_binding_set_t &needed,
	                      column_binding_map_t<ColumnBinding> &remap);
	static void RemapBindings(Expression &expr, const column_binding_map_t<ColumnBinding> &remap);

private:
	Optimizer &optimizer;
};

} // namespace duckdb
