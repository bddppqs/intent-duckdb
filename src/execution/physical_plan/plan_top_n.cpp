#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalTopN &op) {
	D_ASSERT(op.children.size() == 1);
	auto &plan = CreatePlan(*op.children[0]);
	// A code key read through stored translations: the Top-N carries codes; a projection decodes the rows it keeps
	auto decodes = dict_global::PlanLateDecode(plan, op.orders);
	auto &top_n = Make<PhysicalTopN>(decodes.empty() ? op.types : plan.types, std::move(op.orders),
	                                 NumericCast<idx_t>(op.limit), NumericCast<idx_t>(op.offset),
	                                 std::move(op.dynamic_filter), op.estimated_cardinality);
	top_n.children.push_back(plan);
	if (decodes.empty()) {
		return top_n;
	}
	vector<unique_ptr<Expression>> select_list;
	for (idx_t i = 0; i < op.types.size(); i++) {
		optional_ptr<const dict_global::LateDecode> decode;
		for (auto &entry : decodes) {
			if (entry.position == i) {
				decode = &entry;
			}
		}
		if (decode) {
			select_list.push_back(dict_global::LateDecodeExpression(*decode, i, op.types[i]));
		} else {
			select_list.push_back(make_uniq<BoundReferenceExpression>(op.types[i], i));
		}
	}
	auto &projection = Make<PhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
	projection.children.push_back(top_n);
	return projection;
}

} // namespace duckdb
