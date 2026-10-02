//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/aggregate/physical_streaming_first_keys.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/table_filter.hpp"

namespace duckdb {

class FirstKeysGlobalState;
class FirstKeysOperatorState;

//! PhysicalStreamingFirstKeys emits the first k distinct group keys of its input in input order (one thread) and,
//! when k keys have been seen or the input ends, pushes one row-level key filter per key column into the probe-side
//! scan's dynamic filter set (FirstKeysAggregate rewrite). It then finishes its pipeline.
class PhysicalStreamingFirstKeys : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::STREAMING_FIRST_KEYS;

public:
	PhysicalStreamingFirstKeys(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                           vector<unique_ptr<Expression>> groups, unique_ptr<FirstKeysAggregateInfo> info,
	                           idx_t estimated_cardinality);

	//! The group key expressions (bound references into the input chunk)
	vector<unique_ptr<Expression>> groups;
	//! The number of keys to emit
	idx_t k;
	//! The probe-side scan's dynamic filter set (may be null: then no filter is pushed)
	shared_ptr<DynamicTableFilterSet> probe_filters;
	//! Per key column: the probe scan's column index the filter applies to
	vector<idx_t> probe_column_indexes;
	//! Per key column: the probe scan's storage type
	vector<LogicalType> probe_storage_types;

public:
	// Operator interface
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalizeResultType FinalExecute(ExecutionContext &context, DataChunk &chunk, GlobalOperatorState &gstate,
	                                        OperatorState &state) const override;

	bool ParallelOperator() const override {
		return false;
	}
	bool RequiresFinalExecute() const override {
		return true;
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override;

private:
	void PushKeyFilters(FirstKeysGlobalState &gstate, FirstKeysOperatorState &state) const;
};

} // namespace duckdb
