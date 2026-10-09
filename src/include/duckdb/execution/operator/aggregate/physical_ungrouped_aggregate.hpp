//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/aggregate/distinct_aggregate_data.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/execution/radix_partitioned_hashtable.hpp"
#include "duckdb/common/unordered_map.hpp"

namespace duckdb {
class RunAggregateData;
class FusedIntegerAggregate;

//! PhysicalUngroupedAggregate is an aggregate operator that can only perform aggregates without any groups
class PhysicalUngroupedAggregate : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::UNGROUPED_AGGREGATE;

public:
	PhysicalUngroupedAggregate(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                           vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality,
	                           TupleDataValidityType distinct_validity);
	~PhysicalUngroupedAggregate() override;

	//! The aggregates that have to be computed
	vector<unique_ptr<Expression>> aggregates;
	unique_ptr<DistinctAggregateData> distinct_data;
	unique_ptr<DistinctAggregateCollectionInfo> distinct_collection_info;
	//! Run-aware aggregation descriptor shared with the table scan (may be null)
	shared_ptr<RunAggregateData> run_aggregate;
	//! The fused kernel's DISTINCT class: count(DISTINCT x) over one non-NULL integer x, as the (x)-keyed partitioned
	//! dedup of the fused integer aggregate. Null when the gate refuses the plan or kFusedDistinctAggregate is off
	unique_ptr<FusedIntegerAggregate> fused;

public:
	// Source interface
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;
	//! Phase 2's source states while the fused path owns the source; the defaults otherwise
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	ProgressData GetProgress(ClientContext &context, GlobalSourceState &gstate) const override;
	InsertionOrderPreservingMap<string> ExtraSourceParams(GlobalSourceState &gstate,
	                                                      LocalSourceState &lstate) const override;

	bool IsSource() const override {
		return true;
	}
	//! Parallel, and NO_ORDER (so the result collector can go parallel), while fused; the defaults otherwise
	bool ParallelSource() const override;
	OrderPreservationType SourceOrder() const override;
	//! The most rows one GetData call writes into its chunk: one (the finalized states), or STANDARD_VECTOR_SIZE when
	//! the fused kernel may own the source
	idx_t SourceRowBound() const;

public:
	// Sink interface
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	InsertionOrderPreservingMap<string> ParamsToString() const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	bool SinkOrderDependent() const override;

private:
	//! Finalize the distinct aggregates
	SinkFinalizeType FinalizeDistinct(Pipeline &pipeline, Event &event, ClientContext &context,
	                                  GlobalSinkState &gstate) const;
	//! Combine the distinct aggregates
	void CombineDistinct(ExecutionContext &context, OperatorSinkCombineInput &input) const;
	//! Sink the distinct aggregates
	void SinkDistinct(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const;
};

} // namespace duckdb
