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
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {
class LogicalAggregate;
class LogicalGet;
class Optimizer;
namespace dict_global {
class ColumnDictionary;
}

//! FirstKeysAggregate rewrites `LIMIT k` (constant, no OFFSET, no ORDER BY) directly above a `GROUP BY` over a base
//! table scan into the same aggregate over a SEMI join of its input against the first k distinct group keys of a
//! second, single-threaded scan of that input. The first-keys operator pushes a row-level key filter into the probe
//! scan when it has seen k keys (or the input end), so the aggregate only processes the rows of those keys. Any k
//! groups are a valid answer for such a query; every returned group carries its exact aggregate values.
//! A VARCHAR key whose column has stored translations and whose every consumer reads it for its identity only is keyed
//! on its codes (kFirstKeysCodeKeys): both scans read the column codes only and a projection directly above each turns
//! them into INTEGER codes, the first-keys pass, the SEMI join and the aggregate group those, and a projection above
//! the LIMIT decodes the at most k output keys.
class FirstKeysAggregate {
public:
	//! The largest admitted LIMIT
	static constexpr const idx_t MAX_KEYS = 1024;

public:
	explicit FirstKeysAggregate(Optimizer &optimizer);

	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	//! A key read as its codes: the stored translations of its column and its type as a string key
	struct CodeGroup {
		shared_ptr<dict_global::ColumnDictionary> dict;
		LogicalType type;
	};
	//! The decode projection above one rewritten LIMIT and the bindings its parents read through it
	struct CodeDecode {
		optional_ptr<LogicalOperator> projection;
		vector<ReplacementBinding> bindings;
	};

	unique_ptr<LogicalOperator> OptimizeOperator(unique_ptr<LogicalOperator> op);
	bool TryRewrite(LogicalOperator &limit_op, optional_ptr<LogicalOperator> &join, vector<CodeGroup> &code_groups);
	//! Whether group `group_index` of the aggregate over `input` (scan column `column_index` of `get`) is keyed on its
	//! codes: its stored translations, or null
	shared_ptr<dict_global::ColumnDictionary> CodeGroupDictionary(LogicalOperator &input, const LogicalAggregate &aggr,
	                                                              LogicalGet &get, idx_t group_index,
	                                                              idx_t column_index);
	//! The codes projection directly above the scan of the aggregate's input chain: the code columns (indexes into the
	//! scan's column ids) as INTEGER codes, every other column passed through; the chain and the aggregate rebound
	void CodesAboveScan(LogicalAggregate &aggr, LogicalGet &get, const unordered_set<idx_t> &code_columns);
	//! The decode projection over a rewritten LIMIT with code groups (the LIMIT itself without)
	unique_ptr<LogicalOperator> DecodeCodeGroups(unique_ptr<LogicalOperator> limit_op,
	                                             const vector<CodeGroup> &code_groups);
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
	//! The decode projections made, whose parents' references are replaced from the plan root once
	vector<CodeDecode> decodes;
};

} // namespace duckdb
