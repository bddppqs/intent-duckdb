#include "duckdb/optimizer/remove_duplicate_groups.hpp"

#include "duckdb/common/pair.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/scalar/operators.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/remove_unused_columns.hpp"
#include "duckdb/optimizer/statistics_propagator.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/common/tuning_defaults.hpp"

namespace duckdb {

void RemoveDuplicateGroups::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		VisitAggregate(op.Cast<LogicalAggregate>());
		break;
	default:
		break;
	}
	LogicalOperatorVisitor::VisitOperatorExpressions(op);
	LogicalOperatorVisitor::VisitOperatorChildren(op);
}

void RemoveDuplicateGroups::VisitAggregate(LogicalAggregate &aggr) {
	if (!aggr.grouping_functions.empty()) {
		return;
	}

	// If there are multiple grouping sets (ROLLUP/CUBE), we cannot remove duplicate groups
	// because the position of groups matters semantically in ROLLUP(col1, col2, col3),
	// even if col1 and col3 reference the same column binding (e.g., after join column replacement)
	if (aggr.grouping_sets.size() > 1) {
		return;
	}

	auto &groups = aggr.groups;

	column_binding_map_t<idx_t> duplicate_map;
	vector<pair<idx_t, idx_t>> duplicates;
	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		const auto &group = groups[group_idx];
		if (group->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
			continue;
		}
		const auto &colref = group->Cast<BoundColumnRefExpression>();
		const auto &binding = colref.binding;
		const auto it = duplicate_map.find(binding);
		if (it == duplicate_map.end()) {
			duplicate_map.emplace(binding, group_idx);
		} else {
			duplicates.emplace_back(it->second, group_idx);
		}
	}

	if (duplicates.empty()) {
		return;
	}

	// Sort duplicates by max duplicate group idx, because we want to remove groups from the back
	sort(duplicates.begin(), duplicates.end(),
	     [](const pair<idx_t, idx_t> &lhs, const pair<idx_t, idx_t> &rhs) { return lhs.second > rhs.second; });

	// Now we want to remove the duplicates, but this alters the column bindings coming out of the aggregate,
	// so we keep track of how they shift and do another round of column binding replacements
	column_binding_map_t<ColumnBinding> group_binding_map;
	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		group_binding_map.emplace(ColumnBinding(aggr.group_index, group_idx),
		                          ColumnBinding(aggr.group_index, group_idx));
	}

	for (idx_t duplicate_idx = 0; duplicate_idx < duplicates.size(); duplicate_idx++) {
		const auto &duplicate = duplicates[duplicate_idx];
		const auto &remaining_idx = duplicate.first;
		const auto &removed_idx = duplicate.second;

		// Store expression and remove it from groups
		stored_expressions.emplace_back(std::move(groups[removed_idx]));
		groups.erase_at(removed_idx);

		// This optimizer should run before statistics propagation, so this should be empty
		// If it runs after, then group_stats should be updated too
		D_ASSERT(aggr.group_stats.empty());

		// Remove from grouping sets too
		for (auto &grouping_set : aggr.grouping_sets) {
			// Replace removed group with duplicate remaining group
			if (grouping_set.erase(removed_idx) != 0) {
				grouping_set.insert(remaining_idx);
			}

			// Indices shifted: Reinsert groups in the set with group_idx - 1
			vector<idx_t> group_indices_to_reinsert;
			for (auto &entry : grouping_set) {
				if (entry > removed_idx) {
					group_indices_to_reinsert.emplace_back(entry);
				}
			}
			for (const auto group_idx : group_indices_to_reinsert) {
				grouping_set.erase(group_idx);
			}
			for (const auto group_idx : group_indices_to_reinsert) {
				grouping_set.insert(group_idx - 1);
			}
		}

		// Update mapping
		auto it = group_binding_map.find(ColumnBinding(aggr.group_index, removed_idx));
		D_ASSERT(it != group_binding_map.end());
		it->second.column_index = remaining_idx;

		for (auto &map_entry : group_binding_map) {
			auto &new_binding = map_entry.second;
			if (new_binding.column_index > removed_idx) {
				new_binding.column_index--;
			}
		}
	}

	// Replace all references to the old group binding with the new group binding
	for (const auto &map_entry : group_binding_map) {
		auto it = column_references.find(map_entry.first);
		if (it != column_references.end()) {
			for (auto expr : it->second) {
				expr.get().binding = map_entry.second;
			}
		}
	}
}

unique_ptr<Expression> RemoveDuplicateGroups::VisitReplace(BoundColumnRefExpression &expr,
                                                           unique_ptr<Expression> *expr_ptr) {
	// add a column reference
	column_references[expr.binding].push_back(expr);
	return nullptr;
}


namespace {
struct DependentGroupProof {
	bool eligible = false;
	bool callback = false;
	bool bounds = false;
	bool base_retained = false;
	idx_t base_index = DConstants::INVALID_INDEX;
	//! the argument of the function that references the retained key (the other is the constant)
	idx_t column_child = 0;
	hugeint_t min = 0;
	hugeint_t max = 0;
	hugeint_t constant = 0;
	const Expression *expression = nullptr;
};

//! The builtin integer `+` or `-` callback of T (the overflow-checked form the binder picks, or the unchecked form
//! statistics propagation swaps in when it proves no overflow): 1 for `+`, -1 for `-`, 0 for anything else
template <class T>
static int IntegerAddSubtract(void (*target)(DataChunk &, ExpressionState &, Vector &)) {
	if (target == &ScalarFunction::BinaryFunction<T, T, T, AddOperatorOverflowCheck> ||
	    target == &ScalarFunction::BinaryFunction<T, T, T, AddOperator>) {
		return 1;
	}
	if (target == &ScalarFunction::BinaryFunction<T, T, T, SubtractOperatorOverflowCheck> ||
	    target == &ScalarFunction::BinaryFunction<T, T, T, SubtractOperator>) {
		return -1;
	}
	return 0;
}

static int IntegerAddSubtract(PhysicalType type, void (*target)(DataChunk &, ExpressionState &, Vector &)) {
	switch (type) {
	case PhysicalType::INT8:
		return IntegerAddSubtract<int8_t>(target);
	case PhysicalType::INT16:
		return IntegerAddSubtract<int16_t>(target);
	case PhysicalType::INT32:
		return IntegerAddSubtract<int32_t>(target);
	case PhysicalType::INT64:
		return IntegerAddSubtract<int64_t>(target);
	case PhysicalType::UINT8:
		return IntegerAddSubtract<uint8_t>(target);
	case PhysicalType::UINT16:
		return IntegerAddSubtract<uint16_t>(target);
	case PhysicalType::UINT32:
		return IntegerAddSubtract<uint32_t>(target);
	case PhysicalType::UINT64:
		return IntegerAddSubtract<uint64_t>(target);
	default:
		return 0;
	}
}

static const Expression &ResolveDependentGroup(const LogicalAggregate &aggr, const Expression &expr) {
	if (expr.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF || aggr.children.empty() ||
	    aggr.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return expr;
	}
	auto &ref = expr.Cast<BoundColumnRefExpression>();
	auto &projection = aggr.children[0]->Cast<LogicalProjection>();
	if (ref.depth == 0 && ref.binding.table_index == projection.table_index &&
	    ref.binding.column_index < projection.expressions.size()) {
		return *projection.expressions[ref.binding.column_index];
	}
	return expr;
}

static vector<DependentGroupProof> ProveDependentGroups(const LogicalAggregate &aggr) {
	vector<DependentGroupProof> proofs(aggr.groups.size());
	bool ordinary = aggr.grouping_functions.empty() && aggr.grouping_sets.size() <= 1;
	if (ordinary && !aggr.grouping_sets.empty()) {
		auto &set = aggr.grouping_sets[0];
		ordinary = set.size() == aggr.groups.size();
		for (idx_t i = 0; ordinary && i < aggr.groups.size(); i++) {
			ordinary = set.find(i) != set.end();
		}
	}
	bool has_real_key = false;
	for (auto &group : aggr.groups) {
		if (!ResolveDependentGroup(aggr, *group).IsFoldable()) {
			has_real_key = true;
		}
	}
	for (idx_t i = 0; i < aggr.groups.size(); i++) {
		auto &proof = proofs[i];
		auto &expr = ResolveDependentGroup(aggr, *aggr.groups[i]);
		proof.expression = &expr;
		if (!ordinary || !has_real_key || aggr.groups.size() <= 1) {
			continue;
		}
		if (expr.GetExpressionType() == ExpressionType::VALUE_CONSTANT) {
			proof.eligible = true;
			continue;
		}
		// x + k, k + x or x - k over one integer type of at most 64 bits, x a retained group key, k a non-NULL constant
		const auto &type = expr.return_type;
		if (expr.GetExpressionType() != ExpressionType::BOUND_FUNCTION || !type.IsIntegral() ||
		    GetTypeIdSize(type.InternalType()) > sizeof(int64_t)) {
			continue;
		}
		auto &func = expr.Cast<BoundFunctionExpression>();
		if (func.children.size() != 2 || func.children[0]->return_type != type || func.children[1]->return_type != type) {
			continue;
		}
		using callback_t = void (*)(DataChunk &, ExpressionState &, Vector &);
		auto callback = func.function.GetFunctionCallback();
		auto target = callback.target<callback_t>();
		const int sign = target ? IntegerAddSubtract(type.InternalType(), *target) : 0;
		proof.column_child = func.children[0]->GetExpressionType() == ExpressionType::BOUND_COLUMN_REF ? 0 : 1;
		if (sign == 0 || (sign < 0 && proof.column_child != 0) ||
		    func.children[proof.column_child]->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF ||
		    func.children[1 - proof.column_child]->GetExpressionType() != ExpressionType::VALUE_CONSTANT) {
			continue;
		}
		auto &x = func.children[proof.column_child]->Cast<BoundColumnRefExpression>();
		auto &k = func.children[1 - proof.column_child]->Cast<BoundConstantExpression>().value;
		if (x.depth != 0 || k.IsNull()) {
			continue;
		}
		proof.constant = sign > 0 ? k.GetValue<hugeint_t>() : -k.GetValue<hugeint_t>();
		auto builtin = sign > 0 ? AddFunction::GetFunction(type, type) : SubtractFunction::GetFunction(type, type);
		// A copied execution pointer is insufficient: custom initialization or
		// statistics/lifecycle hooks can introduce effects that moving it suppresses.
		proof.callback =
		    !func.bind_info && !func.function.HasInitStateCallback() && !func.function.HasBindCallback() &&
		    !func.function.HasBindExtendedCallback() && !func.function.HasBindLambdaCallback() &&
		    !func.function.HasBindExpressionCallback() && !func.function.HasModifiedDatabasesCallback() &&
		    !func.function.HasExtraFunctionInfo() && !func.function.GetSerializeCallback() &&
		    !func.function.GetDeserializeCallback() &&
		    func.function.GetStatisticsCallback() == builtin.GetStatisticsCallback() &&
		    func.function.GetStability() == builtin.GetStability() &&
		    func.function.GetErrorMode() == builtin.GetErrorMode() &&
		    func.function.arguments == builtin.arguments && func.function.varargs == builtin.varargs &&
		    func.function.GetNullHandling() == FunctionNullHandling::DEFAULT_NULL_HANDLING;
		if (!proof.callback) {
			continue;
		}
		for (idx_t j = 0; j < aggr.groups.size(); j++) {
			auto &base = ResolveDependentGroup(aggr, *aggr.groups[j]);
			if (base.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF || base.return_type != type) {
				continue;
			}
			auto &base_ref = base.Cast<BoundColumnRefExpression>();
			if (base_ref.depth == 0 && base_ref.binding == x.binding) {
				proof.base_retained = true;
				proof.base_index = j;
				break;
			}
		}
		if (!proof.base_retained) {
			continue;
		}
		if (proof.base_index >= aggr.group_stats.size() || !aggr.group_stats[proof.base_index] ||
		    aggr.group_stats[proof.base_index]->GetType() != type ||
		    !NumericStats::HasMinMax(*aggr.group_stats[proof.base_index])) {
			continue;
		}
		// the range proof: no value of x in its statistics' range overflows x + k (or x - k), so the key removed can
		// raise no error the kept key would have raised
		auto &stats = *aggr.group_stats[proof.base_index];
		proof.bounds = true;
		proof.min = NumericStats::Min(stats).GetValue<hugeint_t>();
		proof.max = NumericStats::Max(stats).GetValue<hugeint_t>();
		proof.eligible = proof.min <= proof.max &&
		                 proof.min + proof.constant >= Value::MinimumValue(type).GetValue<hugeint_t>() &&
		                 proof.max + proof.constant <= Value::MaximumValue(type).GetValue<hugeint_t>();
	}
	return proofs;
}
} // namespace

namespace {
static bool RewriteDependentGroups(Optimizer &optimizer, unique_ptr<LogicalOperator> &root,
                                   unique_ptr<LogicalOperator> &node) {
	bool changed = false;
	for (auto &child : node->children) {
		changed = RewriteDependentGroups(optimizer, root, child) || changed;
	}
	if (node->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return changed;
	}
	auto &aggr = node->Cast<LogicalAggregate>();
	auto proofs = ProveDependentGroups(aggr);
	idx_t remove_count = 0;
	for (auto &proof : proofs) {
		remove_count += proof.eligible;
	}
	if (!remove_count) {
		return changed;
	}
	const auto old_count = aggr.groups.size();
	vector<idx_t> retained_index(old_count, DConstants::INVALID_INDEX);
	idx_t remaining = 0;
	for (idx_t i = 0; i < old_count; i++) {
		if (!proofs[i].eligible) {
			retained_index[i] = remaining++;
		}
	}
	D_ASSERT(remaining > 0);
	auto projection_index = optimizer.binder.GenerateTableIndex();
	vector<unique_ptr<Expression>> output;
	ColumnBindingReplacer replacer;
	for (idx_t i = 0; i < old_count; i++) {
		unique_ptr<Expression> expr;
		if (!proofs[i].eligible) {
			expr = make_uniq<BoundColumnRefExpression>(aggr.groups[i]->return_type,
			                                          ColumnBinding(aggr.group_index, retained_index[i]));
		} else {
			expr = proofs[i].expression->Copy();
			if (expr->GetExpressionType() == ExpressionType::BOUND_FUNCTION) {
				auto &func = expr->Cast<BoundFunctionExpression>();
				func.children[proofs[i].column_child] = make_uniq<BoundColumnRefExpression>(
				    func.return_type, ColumnBinding(aggr.group_index, retained_index[proofs[i].base_index]));
			}
		}
		expr->SetAlias(aggr.groups[i]->GetAlias());
		output.push_back(std::move(expr));
		replacer.replacement_bindings.emplace_back(ColumnBinding(aggr.group_index, i),
		                                            ColumnBinding(projection_index, i));
	}
	for (idx_t i = 0; i < aggr.expressions.size(); i++) {
		auto ref = make_uniq<BoundColumnRefExpression>(aggr.expressions[i]->return_type,
		                                              ColumnBinding(aggr.aggregate_index, i));
		ref->SetAlias(aggr.expressions[i]->GetAlias());
		output.push_back(std::move(ref));
		replacer.replacement_bindings.emplace_back(ColumnBinding(aggr.aggregate_index, i),
		                                            ColumnBinding(projection_index, old_count + i));
	}

	vector<unique_ptr<Expression>> groups;
	vector<unique_ptr<BaseStatistics>> group_stats;
	for (idx_t i = 0; i < old_count; i++) {
		if (!proofs[i].eligible) {
			groups.push_back(std::move(aggr.groups[i]));
			if (!aggr.group_stats.empty()) {
				group_stats.push_back(i < aggr.group_stats.size() ? std::move(aggr.group_stats[i]) : nullptr);
			}
		}
	}
	aggr.groups = std::move(groups);
	aggr.group_stats = std::move(group_stats);
	if (!aggr.grouping_sets.empty()) {
		aggr.grouping_sets[0].clear();
		for (idx_t i = 0; i < remaining; i++) {
			aggr.grouping_sets[0].insert(i);
		}
	}
	auto projection = make_uniq<LogicalProjection>(projection_index, std::move(output));
	if (node->has_estimated_cardinality) {
		projection->SetEstimatedCardinality(node->estimated_cardinality);
	}
	// Parent bindings change, but the reconstruction projection intentionally refers
	// to the aggregate's own bindings, which must not be rewritten into themselves.
	replacer.stop_operator = projection.get();
	projection->children.push_back(std::move(node));
	node = std::move(projection);
	replacer.VisitOperator(*root);
	return true;
}
} // namespace

// Run in the early duplicate-groups pass, before compressed materialization and the plan-time aggregate rewrites, so
// they see the retained keys alone and the constant is never compressed into a per-row key. No group_stats exist yet,
// so ProveDependentGroups admits only the constant keys here; x + k keys wait for the pass after statistics propagation.
// Off with kRemoveConstantGroupKeys.
void RemoveDuplicateGroups::RemoveConstantGroups(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan) {
	if (!kRemoveConstantGroupKeys || !RewriteDependentGroups(optimizer, plan, plan)) {
		return;
	}
	RemoveUnusedColumns unused(optimizer);
	unused.VisitOperator(*plan);
}

void RemoveDuplicateGroups::RemoveDependentGroups(
    Optimizer &optimizer, unique_ptr<LogicalOperator> &plan,
    column_binding_map_t<unique_ptr<BaseStatistics>> &statistics_map) {
	if (!RewriteDependentGroups(optimizer, plan, plan)) {
		return;
	}
	// Reuse the existing pruner rather than leaving an obsolete per-row producer.
	// It may renumber projection bindings: rebuild statistics for the resulting
	// tree before any downstream consumer sees the old binding map.
	RemoveUnusedColumns unused(optimizer);
	unused.VisitOperator(*plan);
	StatisticsPropagator propagator(optimizer, *plan);
	propagator.PropagateStatistics(plan);
	statistics_map = propagator.GetStatisticsMap();
}

} // namespace duckdb
