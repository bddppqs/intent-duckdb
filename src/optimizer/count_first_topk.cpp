#include "duckdb/optimizer/count_first_topk.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <cstdlib>

namespace duckdb {

// The count-first top-k rewrite runs when kCountFirstTopK is set; SET disabled_optimizers = 'count_first_topk'
// disables it as every other optimizer.

bool CountFirstTopK::Enabled() {
	return kCountFirstTopK;
}

CountFirstTopK::CountFirstTopK(Optimizer &optimizer) : optimizer(optimizer) {
}

unique_ptr<LogicalOperator> CountFirstTopK::Optimize(unique_ptr<LogicalOperator> op) {
	if (op->type == LogicalOperatorType::LOGICAL_TOP_N && TryRewrite(*op)) {
		// the Top-N now sits over the unchanged aggregate over a SEMI join of the original projection/filter chain and
		// its scan (no Top-N below) against the count-only subtree over a copy of that chain: nothing left to rewrite
		return op;
	}
	for (auto &child : op->children) {
		child = Optimize(std::move(child));
	}
	return op;
}

//! A group key type FirstKeysAggregate admits (its IsFirstKeysPushableType, copied): neither nested nor INTERVAL, and
//! an integer of any width, FLOAT, DOUBLE, VARCHAR or BOOL physical type.
static bool IsPushableKeyType(const LogicalType &type) {
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

static bool IsPlainColumnRef(const Expression &expr) {
	return expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
	       expr.Cast<BoundColumnRefExpression>().depth == 0;
}

//! The ordering count is count_star() or count(column), without DISTINCT, FILTER or ORDER BY
static bool IsOrderingCount(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
		return false;
	}
	auto &aggr = expr.Cast<BoundAggregateExpression>();
	if (aggr.IsDistinct() || aggr.filter || aggr.order_bys) {
		return false;
	}
	if (aggr.function.name == "count_star") {
		return aggr.children.empty();
	}
	if (aggr.function.name == "count") {
		return aggr.children.size() == 1 && IsPlainColumnRef(*aggr.children[0]);
	}
	return false;
}

//! The structural admission of the aggregate (one grouping set over every group, no grouping functions, no
//! first-keys marker, every group a depth-0 column reference of a pushable type) with any bound aggregates
bool CountFirstTopK::AdmitAggregate(const LogicalAggregate &aggr) {
	if (aggr.first_keys || aggr.groups.empty() || aggr.grouping_sets.size() != 1 || !aggr.grouping_functions.empty()) {
		return false;
	}
	if (aggr.grouping_sets[0].size() != aggr.groups.size() || aggr.children.size() != 1) {
		return false;
	}
	for (auto &group : aggr.groups) {
		if (!IsPlainColumnRef(*group) || !IsPushableKeyType(group->return_type)) {
			return false;
		}
	}
	for (auto &expr : aggr.expressions) {
		if (expr->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
	}
	return true;
}

//! FirstKeysAggregate::AdmitGet, copied: DuckDB's own table scan with plain columns and no dynamic filters
bool CountFirstTopK::AdmitGet(const LogicalGet &get) {
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

//! FirstKeysAggregate::ResolveScanColumn, copied
bool CountFirstTopK::ResolveScanColumn(const LogicalOperator &input, ColumnBinding binding, const LogicalGet &get,
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

//! FirstKeysAggregate::CopyInput, copied
unique_ptr<LogicalOperator> CountFirstTopK::CopyInput(const LogicalOperator &input,
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

void CountFirstTopK::ReplaceTableIndices(Expression &expr, const unordered_map<idx_t, idx_t> &index_map) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &colref = expr.Cast<BoundColumnRefExpression>();
		auto entry = index_map.find(colref.binding.table_index);
		if (entry != index_map.end()) {
			colref.binding.table_index = entry->second;
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { ReplaceTableIndices(child, index_map); });
}

//! FirstKeysAggregate::ReplaceTableIndices, copied
void CountFirstTopK::ReplaceTableIndices(LogicalOperator &op, const unordered_map<idx_t, idx_t> &index_map) {
	for (auto &expr : op.expressions) {
		ReplaceTableIndices(*expr, index_map);
	}
	for (auto &child : op.children) {
		ReplaceTableIndices(*child, index_map);
	}
}

static void CollectBindings(Expression &expr, column_binding_set_t &out) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		out.insert(expr.Cast<BoundColumnRefExpression>().binding);
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { CollectBindings(child, out); });
}

void CountFirstTopK::RemapBindings(Expression &expr, const column_binding_map_t<ColumnBinding> &remap) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &colref = expr.Cast<BoundColumnRefExpression>();
		auto entry = remap.find(colref.binding);
		if (entry != remap.end()) {
			colref.binding = entry->second;
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { RemapBindings(child, remap); });
}

//! No later pass prunes a LogicalGet, so the copied scan would read every column of the original one.
//! Keep only what the count pass reads: a projection column nobody above reads becomes a NULL constant of its type
//! (positions, hence bindings, are kept), a filter keeps its predicate's columns and passes every column through, and
//! the scan keeps the columns read above plus the columns its table filters test. `remap` receives the scan's
//! surviving bindings (old -> new position); the caller applies it to every expression above the scan.
void CountFirstTopK::PruneCopy(LogicalOperator &op, const column_binding_set_t &needed,
                               column_binding_map_t<ColumnBinding> &remap) {
	if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = op.Cast<LogicalProjection>();
		column_binding_set_t below;
		for (idx_t i = 0; i < proj.expressions.size(); i++) {
			if (needed.find(ColumnBinding(proj.table_index, i)) == needed.end()) {
				proj.expressions[i] = make_uniq<BoundConstantExpression>(Value(proj.expressions[i]->return_type));
				continue;
			}
			CollectBindings(*proj.expressions[i], below);
		}
		PruneCopy(*op.children[0], below, remap);
		return;
	}
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op.Cast<LogicalFilter>();
		column_binding_set_t below = needed;
		for (auto &expr : filter.expressions) {
			CollectBindings(*expr, below);
		}
		filter.projection_map.clear();
		PruneCopy(*op.children[0], below, remap);
		return;
	}
	D_ASSERT(op.type == LogicalOperatorType::LOGICAL_GET);
	auto &get = op.Cast<LogicalGet>();
	auto &column_ids = get.GetMutableColumnIds();
	vector<ColumnIndex> kept_ids;
	unordered_map<idx_t, idx_t> position;
	for (idx_t i = 0; i < column_ids.size(); i++) {
		bool read_above = needed.find(ColumnBinding(get.table_index, i)) != needed.end();
		bool filtered = get.table_filters.filters.find(column_ids[i].GetPrimaryIndex()) != get.table_filters.filters.end();
		if (read_above || filtered) {
			position[i] = kept_ids.size();
			kept_ids.push_back(column_ids[i]);
		}
	}
	if (kept_ids.empty() || kept_ids.size() == column_ids.size()) {
		return;
	}
	vector<idx_t> projection_ids;
	for (auto id : get.projection_ids) {
		auto entry = position.find(id);
		if (entry != position.end()) {
			projection_ids.push_back(entry->second);
		}
	}
	for (auto &entry : position) {
		remap[ColumnBinding(get.table_index, entry.first)] = ColumnBinding(get.table_index, entry.second);
	}
	column_ids = std::move(kept_ids);
	get.projection_ids = std::move(projection_ids);
}

bool CountFirstTopK::TryRewrite(LogicalOperator &topn_op) {
	auto &topn = topn_op.Cast<LogicalTopN>();
	if (topn.offset != 0 || topn.limit < 1 || topn.limit > MAX_LIMIT || topn.orders.size() != 1 ||
	    topn.children.size() != 1) {
		return false;
	}
	auto &order = topn.orders[0];
	if (order.type != OrderType::DESCENDING || !IsPlainColumnRef(*order.expression)) {
		return false;
	}
	// ... resolved through projections of depth-0 column references only (D5) to one aggregate below
	auto binding = order.expression->Cast<BoundColumnRefExpression>().binding;
	reference<LogicalOperator> current = *topn.children[0];
	while (current.get().type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &proj = current.get().Cast<LogicalProjection>();
		for (auto &expr : proj.expressions) {
			if (!IsPlainColumnRef(*expr)) {
				return false;
			}
		}
		if (binding.table_index != proj.table_index || binding.column_index >= proj.expressions.size() ||
		    proj.children.size() != 1) {
			return false;
		}
		binding = proj.expressions[binding.column_index]->Cast<BoundColumnRefExpression>().binding;
		current = *proj.children[0];
	}
	if (current.get().type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return false;
	}
	auto &aggr = current.get().Cast<LogicalAggregate>();
	if (!AdmitAggregate(aggr) || binding.table_index != aggr.aggregate_index ||
	    binding.column_index >= aggr.expressions.size() || aggr.expressions.size() < 2) {
		return false;
	}
	const auto count_index = binding.column_index;
	if (!IsOrderingCount(*aggr.expressions[count_index])) {
		return false;
	}
	auto &input = *aggr.children[0];
	reference<LogicalOperator> chain = input;
	bool filtered = false;
	while (chain.get().type != LogicalOperatorType::LOGICAL_GET) {
		auto &op = chain.get();
		if (op.type != LogicalOperatorType::LOGICAL_PROJECTION && op.type != LogicalOperatorType::LOGICAL_FILTER) {
			return false;
		}
		filtered = filtered || op.type == LogicalOperatorType::LOGICAL_FILTER;
		if (op.children.size() != 1) {
			return false;
		}
		// the copied chain evaluates every expression independently: a volatile expression could admit a key on the
		// copy that has no surviving row on the original input (the LateMaterialization refusal)
		for (auto &expr : op.expressions) {
			if (expr->IsVolatile()) {
				return false;
			}
		}
		chain = *op.children[0];
	}
	auto &get = chain.get().Cast<LogicalGet>();
	if (!AdmitGet(get)) {
		return false;
	}
	filtered = filtered || !get.table_filters.filters.empty();
	auto &context = optimizer.context;
	vector<ExpressionType> comparisons;
	idx_t key_distinct = 0;
	for (auto &group : aggr.groups) {
		auto &colref = group->Cast<BoundColumnRefExpression>();
		idx_t column_index;
		if (!ResolveScanColumn(input, colref.binding, get, column_index)) {
			return false;
		}
		auto &column_id = get.GetColumnIds()[column_index];
		auto &storage_type = get.GetColumnType(column_id);
		if (storage_type != group->return_type || !IsPushableKeyType(storage_type)) {
			return false;
		}
		// the join's key equality per column, from the scan statistics (FirstKeysAggregate's rule)
		unique_ptr<BaseStatistics> stats;
		if (get.function.statistics_extended) {
			TableFunctionGetStatisticsInput stats_input(get.bind_data.get(), column_id);
			stats = get.function.statistics_extended(context, stats_input);
		} else if (get.function.statistics) {
			stats = get.function.statistics(context, get.bind_data.get(), column_id.GetPrimaryIndex());
		}
		bool no_nulls = stats && !stats->CanHaveNull();
		comparisons.push_back(no_nulls ? ExpressionType::COMPARE_EQUAL : ExpressionType::COMPARE_NOT_DISTINCT_FROM);
		if (stats) {
			key_distinct = MaxValue<idx_t>(key_distinct, stats->GetDistinctCount());
		}
	}
	// the payload aggregates' states (every aggregate but the ordering count) take at least 16 bytes per group: below
	// that, deferring the payload saves less than the second pass costs
	idx_t payload_bytes = 0;
	for (idx_t i = 0; i < aggr.expressions.size(); i++) {
		if (i == count_index) {
			continue;
		}
		auto &function = aggr.expressions[i]->Cast<BoundAggregateExpression>().function;
		if (function.state_size) {
			payload_bytes += function.state_size(function);
		}
	}
	if (payload_bytes < MIN_PAYLOAD_STATE_BYTES) {
		return false;
	}
	const idx_t estimated_groups = aggr.EstimateCardinality(context);
	// an unfiltered input: under a filter neither the row count (a default selectivity guess) nor the key's distinct
	// count (taken over the whole table) describes the filtered rows, so the near-unique test below could not be
	// trusted (refusing is the conservative choice), and the copied chain would filter twice
	if (filtered) {
		return false;
	}
	// at least MIN_ROWS input rows, the fused aggregate's floor: below it the aggregate is small and a second pass does
	// not pay
	const idx_t rows = input.EstimateCardinality(context);
	if (rows < MIN_ROWS) {
		return false;
	}
	// one group key near-unique by its own distinct-count statistic (at most MAX_ROWS_PER_KEY_VALUE rows per value; a
	// composite key has at least as many groups as any of its columns): a full hash aggregate then barely reduces the
	// data, so counting first and finishing only the top-k groups pays off
	if (static_cast<double>(rows) > static_cast<double>(MAX_ROWS_PER_KEY_VALUE) * static_cast<double>(key_distinct)) {
		return false;
	}

	// the count pass: a copy of the input chain with fresh table indices
	unordered_map<idx_t, idx_t> index_map;
	auto copy = CopyInput(input, index_map);
	ReplaceTableIndices(*copy, index_map);
	auto &binder = optimizer.binder;
	auto inner = make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), binder.GenerateTableIndex(),
	                                         vector<unique_ptr<Expression>>());
	GroupingSet grouping_set;
	for (idx_t i = 0; i < aggr.groups.size(); i++) {
		auto group = aggr.groups[i]->Copy();
		ReplaceTableIndices(*group, index_map);
		inner->groups.push_back(std::move(group));
		grouping_set.insert(i);
	}
	inner->grouping_sets.push_back(std::move(grouping_set));
	inner->groupings_index = DConstants::INVALID_INDEX;
	auto count = aggr.expressions[count_index]->Copy();
	ReplaceTableIndices(*count, index_map);
	auto count_type = count->return_type;
	inner->expressions.push_back(std::move(count));
	// the copied scan reads the groups, the count's argument and the chain's filter columns only
	column_binding_set_t needed;
	for (auto &group : inner->groups) {
		CollectBindings(*group, needed);
	}
	CollectBindings(*inner->expressions[0], needed);
	column_binding_map_t<ColumnBinding> remap;
	PruneCopy(*copy, needed, remap);
	if (!remap.empty()) {
		reference<LogicalOperator> node = *copy;
		while (true) {
			for (auto &expr : node.get().expressions) {
				RemapBindings(*expr, remap);
			}
			if (node.get().children.empty()) {
				break;
			}
			node = *node.get().children[0];
		}
		for (auto &group : inner->groups) {
			RemapBindings(*group, remap);
		}
		RemapBindings(*inner->expressions[0], remap);
	}
	inner->SetEstimatedCardinality(estimated_groups);
	inner->children.push_back(std::move(copy));
	// the inner Top-N: the only tie choice among counts
	const auto inner_group_index = inner->group_index;
	vector<BoundOrderByNode> inner_orders;
	inner_orders.emplace_back(OrderType::DESCENDING, order.null_order,
	                          make_uniq<BoundColumnRefExpression>(count_type, ColumnBinding(inner->aggregate_index, 0)));
	auto inner_topn = make_uniq<LogicalTopN>(std::move(inner_orders), topn.limit, 0);
	inner_topn->SetEstimatedCardinality(topn.limit);
	inner_topn->children.push_back(std::move(inner));
	// SEMI join: every row of the original input whose keys are among the top-k keys
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::SEMI);
	for (idx_t i = 0; i < aggr.groups.size(); i++) {
		auto &group = aggr.groups[i];
		JoinCondition condition;
		condition.comparison = comparisons[i];
		condition.left = group->Copy();
		condition.right = make_uniq<BoundColumnRefExpression>(group->GetName(), group->return_type,
		                                                      ColumnBinding(inner_group_index, i));
		join->conditions.push_back(std::move(condition));
	}
	if (input.has_estimated_cardinality) {
		join->SetEstimatedCardinality(input.estimated_cardinality);
	}
	join->children.push_back(std::move(aggr.children[0]));
	join->children.push_back(std::move(inner_topn));
	aggr.children[0] = std::move(join);
	return true;
}

} // namespace duckdb
