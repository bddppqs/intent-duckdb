//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/remove_duplicate_groups.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

namespace duckdb {

class BoundColumnRefExpression;
class BaseStatistics;
class Optimizer;

//! The RemoveDuplicateGroups optimizer traverses the logical operator tree and removes any duplicate aggregate groups
//! Duplicate groups may be introduced when joins columns are removed, e.g., by Deliminator or RemoveUnusedColumns
class RemoveDuplicateGroups : public LogicalOperatorVisitor {
public:
	RemoveDuplicateGroups() {
	}

	void VisitOperator(LogicalOperator &op) override;
	static void RemoveDependentGroups(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan,
	                                  column_binding_map_t<unique_ptr<BaseStatistics>> &statistics_map);

private:
	void VisitAggregate(LogicalAggregate &aggr);

protected:
	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override;

private:
	//! The map of column references
	column_binding_map_t<vector<reference<BoundColumnRefExpression>>> column_references;
	//! Stored expressions (kept around so we don't have dangling pointers)
	vector<unique_ptr<Expression>> stored_expressions;
};

} // namespace duckdb
