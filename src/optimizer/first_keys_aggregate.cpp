#include "duckdb/optimizer/first_keys_aggregate.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

namespace duckdb {

FirstKeysAggregate::FirstKeysAggregate(Optimizer &optimizer) : optimizer(optimizer) {
}

unique_ptr<LogicalOperator> FirstKeysAggregate::Optimize(unique_ptr<LogicalOperator> op) {
	if (op->type == LogicalOperatorType::LOGICAL_LIMIT && TryRewrite(*op)) {
		// op is now LIMIT -> AGGREGATE -> SEMI JOIN(original input, first-keys subtree): only the original input
		// can hold further candidates
		auto &join = *op->children[0]->children[0];
		join.children[0] = Optimize(std::move(join.children[0]));
		return op;
	}
	for (auto &child : op->children) {
		child = Optimize(std::move(child));
	}
	return op;
}

//! A group key type the pushed row-level key filter can evaluate: neither nested nor INTERVAL (the same refusal as
//! JoinFilterPushdownOptimizer), and a physical type ColumnSegment::FilterSelection handles for CONSTANT_COMPARISON
//! (integers of every width, FLOAT, DOUBLE, VARCHAR, BOOL). Any other type is not rewritten.
static bool IsFirstKeysPushableType(const LogicalType &type) {
	if (type.IsNested() || type.id() == LogicalTypeId::INTERVAL) {
		return false;
	}
	switch (type.InternalType()) {
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::INT128:
	case PhysicalType::UINT128:
	case PhysicalType::FLOAT:
	case PhysicalType::DOUBLE:
	case PhysicalType::VARCHAR:
	case PhysicalType::BOOL:
		return true;
	default:
		return false;
	}
}

static bool IsAdmittedAggregate(const BoundAggregateExpression &aggr) {
	if (aggr.IsDistinct() || aggr.filter || aggr.order_bys) {
		return false;
	}
	auto &name = aggr.function.name;
	return name == "count_star" || name == "count" || name == "sum" || name == "min" || name == "max" || name == "avg";
}

bool FirstKeysAggregate::AdmitAggregate(const LogicalAggregate &aggr) const {
	if (aggr.first_keys || aggr.groups.empty() || aggr.grouping_sets.size() != 1 || !aggr.grouping_functions.empty()) {
		return false;
	}
	if (aggr.grouping_sets[0].size() != aggr.groups.size()) {
		return false;
	}
	for (auto &group : aggr.groups) {
		if (group->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
			return false;
		}
		if (group->Cast<BoundColumnRefExpression>().depth != 0) {
			return false;
		}
		if (!IsFirstKeysPushableType(group->return_type)) {
			return false;
		}
	}
	for (auto &expr : aggr.expressions) {
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		if (!IsAdmittedAggregate(expr->Cast<BoundAggregateExpression>())) {
			return false;
		}
	}
	return true;
}

bool FirstKeysAggregate::AdmitGet(const LogicalGet &get) {
	if (get.function.name != "seq_scan" || !get.GetTable() || !get.function.filter_pushdown ||
	    !get.function.projection_pushdown) {
		return false;
	}
	if (!get.children.empty() || !get.projected_input.empty() || get.ordinality_idx.IsValid() ||
	    get.row_group_order_options || get.extra_info.sample_options || get.dynamic_filters) {
		return false;
	}
	if (get.GetColumnIds().empty()) {
		return false;
	}
	for (auto &column_id : get.GetColumnIds()) {
		if (column_id.IsVirtualColumn() || column_id.HasChildren()) {
			return false;
		}
	}
	return true;
}

bool FirstKeysAggregate::ResolveScanColumn(const LogicalOperator &input, ColumnBinding binding, const LogicalGet &get,
                                           idx_t &column_index) {
	reference<const LogicalOperator> current = input;
	while (current.get().type != LogicalOperatorType::LOGICAL_GET) {
		auto &op = current.get();
		if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = op.Cast<LogicalProjection>();
			if (binding.table_index != proj.table_index || binding.column_index >= proj.expressions.size()) {
				return false;
			}
			auto &expr = *proj.expressions[binding.column_index];
			if (expr.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				return false;
			}
			binding = expr.Cast<BoundColumnRefExpression>().binding;
		} else if (op.type != LogicalOperatorType::LOGICAL_FILTER) {
			return false;
		}
		current = *op.children[0];
	}
	if (binding.table_index != get.table_index || binding.column_index >= get.GetColumnIds().size()) {
		return false;
	}
	column_index = binding.column_index;
	return true;
}

unique_ptr<LogicalOperator> FirstKeysAggregate::CopyInput(const LogicalOperator &input,
                                                          unordered_map<idx_t, idx_t> &index_map) {
	auto &binder = optimizer.binder;
	if (input.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = input.Cast<LogicalGet>();
		auto new_get = make_uniq<LogicalGet>(binder.GenerateTableIndex(), get.function, get.bind_data->Copy(),
		                                     get.returned_types, get.names, get.virtual_columns);
		new_get->GetMutableColumnIds() = get.GetColumnIds();
		new_get->projection_ids = get.projection_ids;
		new_get->parameters = get.parameters;
		new_get->named_parameters = get.named_parameters;
		new_get->input_table_types = get.input_table_types;
		new_get->input_table_names = get.input_table_names;
		for (auto &entry : get.table_filters.filters) {
			new_get->table_filters.filters[entry.first] = entry.second->Copy();
		}
		if (get.has_estimated_cardinality) {
			new_get->SetEstimatedCardinality(get.estimated_cardinality);
		}
		index_map[get.table_index] = new_get->table_index;
		return std::move(new_get);
	}
	auto child = CopyInput(*input.children[0], index_map);
	vector<unique_ptr<Expression>> expressions;
	for (auto &expr : input.expressions) {
		expressions.push_back(expr->Copy());
	}
	unique_ptr<LogicalOperator> result;
	if (input.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = input.Cast<LogicalProjection>();
		auto new_proj = make_uniq<LogicalProjection>(binder.GenerateTableIndex(), std::move(expressions));
		index_map[proj.table_index] = new_proj->table_index;
		result = std::move(new_proj);
	} else {
		D_ASSERT(input.type == LogicalOperatorType::LOGICAL_FILTER);
		auto &filter = input.Cast<LogicalFilter>();
		auto new_filter = make_uniq<LogicalFilter>();
		new_filter->expressions = std::move(expressions);
		new_filter->projection_map = filter.projection_map;
		result = std::move(new_filter);
	}
	if (input.has_estimated_cardinality) {
		result->SetEstimatedCardinality(input.estimated_cardinality);
	}
	result->children.push_back(std::move(child));
	return result;
}

void FirstKeysAggregate::ReplaceTableIndices(LogicalOperator &op, const unordered_map<idx_t, idx_t> &index_map) {
	for (auto &expr : op.expressions) {
		ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
		    expr, [&](BoundColumnRefExpression &colref, unique_ptr<Expression> &) {
			    auto entry = index_map.find(colref.binding.table_index);
			    if (entry != index_map.end()) {
				    colref.binding.table_index = entry->second;
			    }
		    });
	}
	for (auto &child : op.children) {
		ReplaceTableIndices(*child, index_map);
	}
}

bool FirstKeysAggregate::TryRewrite(LogicalOperator &limit_op) {
	auto &limit = limit_op.Cast<LogicalLimit>();
	if (limit.limit_val.Type() != LimitNodeType::CONSTANT_VALUE || limit.offset_val.Type() != LimitNodeType::UNSET) {
		return false;
	}
	auto k = limit.limit_val.GetConstantValue();
	if (k < 1 || k > MAX_KEYS) {
		return false;
	}
	if (limit_op.children.size() != 1 ||
	    limit_op.children[0]->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &aggr = limit_op.children[0]->Cast<LogicalAggregate>();
	if (!AdmitAggregate(aggr) || aggr.children.size() != 1) {
		return false;
	}
	// the aggregate input: a base-table scan under zero or more projections/filters
	auto &input = *aggr.children[0];
	reference<LogicalOperator> current = input;
	while (current.get().type != LogicalOperatorType::LOGICAL_GET) {
		auto &op = current.get();
		if (op.type != LogicalOperatorType::LOGICAL_PROJECTION && op.type != LogicalOperatorType::LOGICAL_FILTER) {
			return false;
		}
		if (op.children.size() != 1) {
			return false;
		}
		// the copied chain evaluates every expression independently: a volatile expression (random()) could admit
		// a key on the copy that has no surviving row on the original input (the LateMaterialization refusal)
		for (auto &expr : op.expressions) {
			if (expr->IsVolatile()) {
				return false;
			}
		}
		current = *op.children[0];
	}
	auto &get = current.get().Cast<LogicalGet>();
	if (!AdmitGet(get)) {
		return false;
	}
	// every group must be a plain column of the scan; decide the key equality per column from the scan statistics
	auto &context = optimizer.context;
	auto info = make_uniq<FirstKeysAggregateInfo>();
	info->k = k;
	vector<ExpressionType> comparisons;
	for (auto &group : aggr.groups) {
		auto &colref = group->Cast<BoundColumnRefExpression>();
		idx_t column_index;
		if (!ResolveScanColumn(input, colref.binding, get, column_index)) {
			return false;
		}
		auto &column_id = get.GetColumnIds()[column_index];
		auto &storage_type = get.GetColumnType(column_id);
		if (storage_type != group->return_type || !IsFirstKeysPushableType(storage_type)) {
			return false;
		}
		unique_ptr<BaseStatistics> stats;
		if (get.function.statistics_extended) {
			TableFunctionGetStatisticsInput stats_input(get.bind_data.get(), column_id);
			stats = get.function.statistics_extended(context, stats_input);
		} else if (get.function.statistics) {
			stats = get.function.statistics(context, get.bind_data.get(), column_id.GetPrimaryIndex());
		}
		bool no_nulls = stats && !stats->CanHaveNull();
		comparisons.push_back(no_nulls ? ExpressionType::COMPARE_EQUAL : ExpressionType::COMPARE_NOT_DISTINCT_FROM);
		info->probe_column_indexes.push_back(column_index);
		info->probe_storage_types.push_back(storage_type);
	}
	// the second scan: a copy of the input chain with fresh table indices
	unordered_map<idx_t, idx_t> index_map;
	auto copy = CopyInput(input, index_map);
	ReplaceTableIndices(*copy, index_map);
	// the marked first-keys aggregate over the copy, under its own LIMIT k
	auto &binder = optimizer.binder;
	auto keys = make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), binder.GenerateTableIndex(),
	                                        vector<unique_ptr<Expression>>());
	GroupingSet grouping_set;
	for (idx_t i = 0; i < aggr.groups.size(); i++) {
		auto &colref = aggr.groups[i]->Cast<BoundColumnRefExpression>();
		auto entry = index_map.find(colref.binding.table_index);
		if (entry == index_map.end()) {
			return false;
		}
		keys->groups.push_back(make_uniq<BoundColumnRefExpression>(
		    colref.GetName(), colref.return_type, ColumnBinding(entry->second, colref.binding.column_index)));
		grouping_set.insert(i);
	}
	keys->grouping_sets.push_back(std::move(grouping_set));
	keys->groupings_index = DConstants::INVALID_INDEX;
	keys->SetEstimatedCardinality(k);
	keys->children.push_back(std::move(copy));
	// the probe-side dynamic filter set lives on the original scan; the first-keys operator pushes into it
	if (!get.dynamic_filters) {
		get.dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
	}
	info->probe_filters = get.dynamic_filters;
	auto keys_index = keys->group_index;
	auto &keys_ref = *keys;
	auto inner_limit = make_uniq<LogicalLimit>(BoundLimitNode::ConstantValue(NumericCast<int64_t>(k)), BoundLimitNode());
	inner_limit->SetEstimatedCardinality(k);
	inner_limit->children.push_back(std::move(keys));
	// SEMI join: the original input against the first k keys
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::SEMI);
	for (idx_t i = 0; i < aggr.groups.size(); i++) {
		auto &group = aggr.groups[i];
		JoinCondition condition;
		condition.comparison = comparisons[i];
		condition.left = group->Copy();
		condition.right =
		    make_uniq<BoundColumnRefExpression>(group->GetName(), group->return_type, ColumnBinding(keys_index, i));
		join->conditions.push_back(std::move(condition));
	}
	keys_ref.first_keys = std::move(info);
	if (input.has_estimated_cardinality) {
		join->SetEstimatedCardinality(input.estimated_cardinality);
	}
	join->children.push_back(std::move(aggr.children[0]));
	join->children.push_back(std::move(inner_limit));
	aggr.children[0] = std::move(join);
	return true;
}

} // namespace duckdb
