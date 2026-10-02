//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/first_keys_aggregate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
class LogicalAggregate;
class LogicalGet;
class Optimizer;

//! FirstKeysAggregate rewrites `LIMIT k` (constant, no OFFSET, no ORDER BY) directly above a `GROUP BY` over a base
//! table scan into the same aggregate over a SEMI join of its input against the first k distinct group keys of a
//! second, single-threaded scan of that input. The first-keys operator pushes a row-level key filter into the probe
//! scan when it has seen k keys (or the input end), so the aggregate only processes the rows of those keys. Any k
//! groups are a valid answer for such a query; every returned group carries its exact aggregate values.
class FirstKeysAggregate {
public:
	//! The largest admitted LIMIT
	static constexpr const idx_t MAX_KEYS = 1024;

public:
	explicit FirstKeysAggregate(Optimizer &optimizer);

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	bool TryRewrite(LogicalOperator &limit_op);
	bool AdmitAggregate(const LogicalAggregate &aggr) const;
	static bool AdmitGet(const LogicalGet &get);
	//! Copy the projection/filter chain and its scan with fresh table indices; fills old -> new table index map
	unique_ptr<LogicalOperator> CopyInput(const LogicalOperator &input, unordered_map<idx_t, idx_t> &index_map);
	static void ReplaceTableIndices(LogicalOperator &op, const unordered_map<idx_t, idx_t> &index_map);
	//! Resolve a group column reference through the chain down to the scan's column index (index into column_ids)
	static bool ResolveScanColumn(const LogicalOperator &input, ColumnBinding binding, const LogicalGet &get,
	                              idx_t &column_index);

private:
	Optimizer &optimizer;
};

} // namespace duckdb
