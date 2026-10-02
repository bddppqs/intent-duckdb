//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_aggregate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/enums/tuple_data_layout_enums.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/parser/group_by_node.hpp"

namespace duckdb {
class DynamicTableFilterSet;

//! Plan-time marker of the first-keys rewrite (FirstKeysAggregate): the marked aggregate emits the first k distinct
//! group keys of its input and pushes a row-level key filter into the probe-side scan. Not serialized: a copy made
//! through serialization is a plain aggregate again (still exact under its LIMIT).
struct FirstKeysAggregateInfo {
	//! The number of keys to emit
	idx_t k = 0;
	//! The probe-side scan's dynamic filter set
	shared_ptr<DynamicTableFilterSet> probe_filters;
	//! Per group: the probe scan's column index (index into its column ids) the key filter applies to
	vector<idx_t> probe_column_indexes;
	//! Per group: the probe scan's storage type of that column
	vector<LogicalType> probe_storage_types;
};

//! LogicalAggregate represents an aggregate operation with (optional) GROUP BY
//! operator.
class LogicalAggregate : public LogicalOperator {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY;

public:
	LogicalAggregate(idx_t group_index, idx_t aggregate_index, vector<unique_ptr<Expression>> select_list);

	//! The table index for the groups of the LogicalAggregate
	idx_t group_index;
	//! The table index for the aggregates of the LogicalAggregate
	idx_t aggregate_index;
	//! The table index for the GROUPING function calls of the LogicalAggregate
	idx_t groupings_index;
	//! The set of groups (optional).
	vector<unique_ptr<Expression>> groups;
	//! The set of grouping sets (optional).
	vector<GroupingSet> grouping_sets;
	//! The list of grouping function calls (optional)
	vector<unsafe_vector<idx_t>> grouping_functions;
	//! Group statistics (optional)
	vector<unique_ptr<BaseStatistics>> group_stats;
	//! Whether the inputs to all expression are non-NULL
	TupleDataValidityType distinct_validity;
	//! First-keys marker (optional, not serialized)
	unique_ptr<FirstKeysAggregateInfo> first_keys;

public:
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	vector<ColumnBinding> GetColumnBindings() override;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);
	idx_t EstimateCardinality(ClientContext &context) override;
	vector<idx_t> GetTableIndex() const override;
	string GetName() const override;

protected:
	void ResolveTypes() override;
};
} // namespace duckdb
