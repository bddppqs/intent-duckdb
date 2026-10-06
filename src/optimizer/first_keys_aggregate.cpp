#include "duckdb/optimizer/first_keys_aggregate.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
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
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"
#include "duckdb/storage/compression/dict_global/persisted_translation.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/transaction/local_storage.hpp"

namespace duckdb {

FirstKeysAggregate::FirstKeysAggregate(Optimizer &optimizer) : optimizer(optimizer) {
}

unique_ptr<LogicalOperator> FirstKeysAggregate::Optimize(unique_ptr<LogicalOperator> op) {
	auto result = OptimizeOperator(std::move(op));
	// a rewritten LIMIT with code groups is read through the decode projection above it: its parents' references to
	// the aggregate's outputs move to that projection's, once from the root (the projection's own references stay)
	for (auto &decode : decodes) {
		ColumnBindingReplacer replacer;
		replacer.replacement_bindings = std::move(decode.bindings);
		replacer.stop_operator = decode.projection;
		replacer.VisitOperator(*result);
	}
	decodes.clear();
	return result;
}

unique_ptr<LogicalOperator> FirstKeysAggregate::OptimizeOperator(unique_ptr<LogicalOperator> op) {
	optional_ptr<LogicalOperator> join;
	vector<CodeGroup> code_groups;
	if (op->type == LogicalOperatorType::LOGICAL_LIMIT && TryRewrite(*op, join, code_groups)) {
		// op is now LIMIT -> AGGREGATE -> SEMI JOIN(original input, first-keys subtree): only the original input
		// can hold further candidates
		join->children[0] = OptimizeOperator(std::move(join->children[0]));
		return DecodeCodeGroups(std::move(op), code_groups);
	}
	for (auto &child : op->children) {
		child = OptimizeOperator(std::move(child));
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

static bool ReferencesAny(const Expression &expr, const column_binding_set_t &bindings) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		return bindings.find(expr.Cast<BoundColumnRefExpression>().binding) != bindings.end();
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		found = found || ReferencesAny(child, bindings);
	});
	return found;
}

shared_ptr<dict_global::ColumnDictionary> FirstKeysAggregate::CodeGroupDictionary(LogicalOperator &input,
                                                                                const LogicalAggregate &aggr,
                                                                                LogicalGet &get, idx_t group_index,
                                                                                idx_t column_index) {
	auto &context = optimizer.context;
	auto &type = aggr.groups[group_index]->return_type;
	if (type.id() != LogicalTypeId::VARCHAR || !StringType::GetCollation(type).empty()) {
		return nullptr;
	}
	auto table = get.GetTable();
	if (!table || !table->IsDuckTable()) {
		return nullptr;
	}
	auto &storage = table->Cast<DuckTableEntry>().GetStorage();
	// the code keys' cost rule (kStoredCodeKeysMinScanRows), read here before statistics propagation; this
	// transaction's local rows hold strings (PlanCodeKeys's refusals)
	if (get.EstimateCardinality(context) < idx_t(kStoredCodeKeysMinScanRows) ||
	    LocalStorage::Get(context, storage.db).Find(storage)) {
		return nullptr;
	}
	auto &column_id = get.GetColumnIds()[column_index];
	if (!column_id.HasPrimaryIndex()) {
		return nullptr;
	}
	// a pushed filter on the column must be decided on codes (a logical get keys its filters by the table's column
	// index; the physical plan maps them to the scan's column ids)
	auto filter = get.table_filters.filters.find(column_id.GetPrimaryIndex());
	if (filter != get.table_filters.filters.end() && !dict_global::CodeTranslatable(*filter->second)) {
		return nullptr;
	}
	// identity only: from the scan up, every filter and every computed projection expression references the column
	// nowhere, a projection only passes it through, and no aggregate reads it
	vector<reference<LogicalOperator>> chain;
	for (reference<LogicalOperator> current = input; current.get().type != LogicalOperatorType::LOGICAL_GET;
	     current = *current.get().children[0]) {
		chain.push_back(current);
	}
	column_binding_set_t carried;
	carried.insert(ColumnBinding(get.table_index, column_index));
	for (idx_t i = chain.size(); i > 0; i--) {
		auto &op = chain[i - 1].get();
		if (op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
			for (auto &expr : op.expressions) {
				if (ReferencesAny(*expr, carried)) {
					return nullptr;
				}
			}
			continue;
		}
		auto &projection = op.Cast<LogicalProjection>();
		column_binding_set_t next;
		for (idx_t e = 0; e < projection.expressions.size(); e++) {
			auto &expr = *projection.expressions[e];
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				if (carried.find(expr.Cast<BoundColumnRefExpression>().binding) != carried.end()) {
					next.insert(ColumnBinding(projection.table_index, e));
				}
			} else if (ReferencesAny(expr, carried)) {
				return nullptr;
			}
		}
		carried = std::move(next);
	}
	for (auto &expr : aggr.expressions) {
		if (ReferencesAny(*expr, carried)) {
			return nullptr;
		}
	}
	auto &column = table->GetColumns().GetColumn(LogicalIndex(column_id.GetPrimaryIndex()));
	return dict_global::PublishPersisted(storage, column.StorageOid());
}

unique_ptr<LogicalOperator> FirstKeysAggregate::DecodeCodeGroups(unique_ptr<LogicalOperator> limit_op,
                                                                 const vector<CodeGroup> &code_groups) {
	bool any_code = false;
	for (auto &code_group : code_groups) {
		any_code = any_code || code_group.dict;
	}
	if (!any_code) {
		return limit_op;
	}
	// LIMIT -> AGGREGATE: the aggregate's outputs, each code group's codes decoded to its string (at most k)
	auto &aggr = limit_op->children[0]->Cast<LogicalAggregate>();
	auto projection_index = optimizer.binder.GenerateTableIndex();
	vector<unique_ptr<Expression>> expressions;
	vector<ReplacementBinding> bindings;
	for (idx_t g = 0; g < aggr.groups.size(); g++) {
		auto &group = *aggr.groups[g];
		ColumnBinding binding(aggr.group_index, g);
		unique_ptr<Expression> expr = make_uniq<BoundColumnRefExpression>(group.GetName(), group.return_type, binding);
		if (code_groups[g].dict) {
			expr = dict_global::CodeDecodeExpression(code_groups[g].dict, std::move(expr), code_groups[g].type);
		}
		expressions.push_back(std::move(expr));
		bindings.emplace_back(binding, ColumnBinding(projection_index, g));
	}
	for (idx_t j = 0; j < aggr.expressions.size(); j++) {
		auto &aggregate = *aggr.expressions[j];
		ColumnBinding binding(aggr.aggregate_index, j);
		expressions.push_back(make_uniq<BoundColumnRefExpression>(aggregate.GetName(), aggregate.return_type, binding));
		bindings.emplace_back(binding, ColumnBinding(projection_index, aggr.groups.size() + j));
	}
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(expressions));
	if (limit_op->has_estimated_cardinality) {
		projection->SetEstimatedCardinality(limit_op->estimated_cardinality);
	}
	projection->children.push_back(std::move(limit_op));
	projection->ResolveOperatorTypes();
	decodes.push_back(CodeDecode {projection.get(), std::move(bindings)});
	return std::move(projection);
}

bool FirstKeysAggregate::TryRewrite(LogicalOperator &limit_op, optional_ptr<LogicalOperator> &join_out,
                                    vector<CodeGroup> &code_groups) {
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
	// the table indexes the copy of the chain maps (its projections' and its scan's)
	unordered_set<idx_t> chain_indexes;
	while (current.get().type != LogicalOperatorType::LOGICAL_GET) {
		auto &op = current.get();
		if (op.type != LogicalOperatorType::LOGICAL_PROJECTION && op.type != LogicalOperatorType::LOGICAL_FILTER) {
			return false;
		}
		if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
			chain_indexes.insert(op.Cast<LogicalProjection>().table_index);
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
	chain_indexes.insert(get.table_index);
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
		// every group's binding is one the copy of the chain maps (checked before any mark below)
		if (chain_indexes.find(colref.binding.table_index) == chain_indexes.end()) {
			return false;
		}
	}
	// the code groups (kFirstKeysCodeKeys): a VARCHAR key read for its identity only, its column with stored
	// translations, is marked codes only on the scan's publication once, before the copy below shares that publication
	// (PublishingTableScanBindData::Copy), so both scans read its codes from the one mark. Off under query
	// verification: the plan's serializer round trip cannot rebuild the two internal functions, and the standard plan
	// is exact
	auto &config = ClientConfig::GetConfig(context);
	code_groups.assign(aggr.groups.size(), CodeGroup());
	unordered_set<idx_t> code_columns;
	if (kFirstKeysCodeKeys && dict_global::PersistedTranslationsEnabled() && dict_global::CodeKeysEnabled() &&
	    !config.query_verification_enabled && !config.verify_serializer) {
		for (idx_t g = 0; g < aggr.groups.size(); g++) {
			auto dict = CodeGroupDictionary(input, aggr, get, g, info->probe_column_indexes[g]);
			if (!dict || !dict_global::MarkCodeGroupKey(context, get, info->probe_column_indexes[g], dict)) {
				continue;
			}
			code_groups[g].dict = std::move(dict);
			code_groups[g].type = aggr.groups[g]->return_type;
			code_columns.insert(info->probe_column_indexes[g]);
			// the first-keys pass pushes no key filter for a code group (its keys are codes, the scan reads codes)
			info->probe_column_indexes[g] = DConstants::INVALID_INDEX;
			info->probe_storage_types[g] = LogicalType::INTEGER;
		}
	}
	auto &binder = optimizer.binder;
	if (!code_columns.empty()) {
		CodesAboveScan(aggr, get, code_columns);
	}
	// the second scan: a copy of the input chain (with the codes projection) with fresh table indices
	auto &rewritten_input = *aggr.children[0];
	unordered_map<idx_t, idx_t> index_map;
	auto copy = CopyInput(rewritten_input, index_map);
	ReplaceTableIndices(*copy, index_map);
	// the marked first-keys aggregate over the copy, under its own LIMIT k
	auto keys = make_uniq<LogicalAggregate>(binder.GenerateTableIndex(), binder.GenerateTableIndex(),
	                                        vector<unique_ptr<Expression>>());
	GroupingSet grouping_set;
	for (idx_t i = 0; i < aggr.groups.size(); i++) {
		auto &colref = aggr.groups[i]->Cast<BoundColumnRefExpression>();
		auto entry = index_map.find(colref.binding.table_index);
		if (entry == index_map.end()) {
			throw InternalException("FirstKeysAggregate: a group the copy of its input does not map");
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
	// SEMI join: the original input against the first k keys (a code group's condition on its codes - the comparison
	// unchanged, NULL is code 0 is NULL)
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
	if (rewritten_input.has_estimated_cardinality) {
		join->SetEstimatedCardinality(rewritten_input.estimated_cardinality);
	}
	join->children.push_back(std::move(aggr.children[0]));
	join->children.push_back(std::move(inner_limit));
	join_out = join.get();
	aggr.children[0] = std::move(join);
	if (!code_columns.empty()) {
		aggr.ResolveOperatorTypes();
	}
	return true;
}

void FirstKeysAggregate::CodesAboveScan(LogicalAggregate &aggr, LogicalGet &get,
                                        const unordered_set<idx_t> &code_columns) {
	// the slot holding the scan, below the chain
	reference<unique_ptr<LogicalOperator>> slot = aggr.children[0];
	vector<reference<LogicalOperator>> chain;
	while (slot.get()->type != LogicalOperatorType::LOGICAL_GET) {
		chain.push_back(*slot.get());
		slot = slot.get()->children[0];
	}
	// a projection directly above the scan: every column passed through, each code group's column as its codes - the
	// codes-only vector is read by this projection alone and never reaches an operator that caches or copies its rows
	auto bindings = get.GetColumnBindings();
	get.ResolveOperatorTypes();
	auto projection_index = optimizer.binder.GenerateTableIndex();
	vector<unique_ptr<Expression>> expressions;
	column_binding_map_t<ColumnBinding> moved;
	column_binding_set_t carried;
	for (idx_t i = 0; i < bindings.size(); i++) {
		unique_ptr<Expression> expr = make_uniq<BoundColumnRefExpression>(get.types[i], bindings[i]);
		if (bindings[i].table_index == get.table_index && code_columns.count(bindings[i].column_index)) {
			expr = dict_global::CodesExpression(std::move(expr));
			carried.insert(ColumnBinding(projection_index, i));
		}
		expressions.push_back(std::move(expr));
		moved[bindings[i]] = ColumnBinding(projection_index, i);
	}
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(expressions));
	if (get.has_estimated_cardinality) {
		projection->SetEstimatedCardinality(get.estimated_cardinality);
	}
	projection->children.push_back(std::move(slot.get()));
	slot.get() = std::move(projection);
	// the chain above it and the aggregate read the projection's outputs; a reference to a code column, a plain
	// pass-through by construction (identity only), becomes INTEGER, and so does every projection output carrying it
	auto rebind = [&](unique_ptr<Expression> &expr) {
		ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
		    expr, [&](BoundColumnRefExpression &colref, unique_ptr<Expression> &) {
			    auto entry = moved.find(colref.binding);
			    if (entry != moved.end()) {
				    colref.binding = entry->second;
			    }
			    if (carried.find(colref.binding) != carried.end()) {
				    colref.return_type = LogicalType::INTEGER;
			    }
		    });
	};
	for (idx_t i = chain.size(); i > 0; i--) {
		auto &op = chain[i - 1].get();
		for (auto &expr : op.expressions) {
			rebind(expr);
		}
		if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &chain_projection = op.Cast<LogicalProjection>();
			for (idx_t e = 0; e < chain_projection.expressions.size(); e++) {
				auto &expr = *chain_projection.expressions[e];
				if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
				    carried.find(expr.Cast<BoundColumnRefExpression>().binding) != carried.end()) {
					carried.insert(ColumnBinding(chain_projection.table_index, e));
				}
			}
		}
	}
	for (auto &group : aggr.groups) {
		rebind(group);
	}
	for (auto &expr : aggr.expressions) {
		rebind(expr);
	}
	aggr.children[0]->ResolveOperatorTypes();
}

} // namespace duckdb
