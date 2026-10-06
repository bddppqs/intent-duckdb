#include "duckdb/optimizer/topn_optimizer.hpp"

#include "duckdb/common/limits.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/filter/selectivity_optional_filter.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/optimizer/join_filter_pushdown_optimizer.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <cstdlib>

namespace duckdb {

// The Top-N bound applied row by row: on the restricted path below, where a row the bound rejects provably never
// reaches the heap, the bound is wrapped in an adaptive SelectivityOptionalFilter that the scan evaluates per vector
// instead of a plain OptionalFilter used for zonemap checks only (kTopNRowwiseBound; off,
// the plain wrapper everywhere).
static bool RowwiseTopNBoundEnabled() {
	return kTopNRowwiseBound;
}

//! The LogicalGet reached from the Top-N's child through projections and filters only, when it is DuckDB's own table
//! scan: a projection of column references and a filter only drop rows or columns, so every row a bound rejects reaches
//! the heap with the same order value and is rejected there too. Any other operator on the path (a join, a cross
//! product, LIMIT, ORDER BY, TOP_N, DISTINCT, UNNEST, a set operation) or any other scan function yields nothing.
static optional_ptr<LogicalGet> RowwiseTopNScan(LogicalOperator &child) {
	reference<LogicalOperator> current = child;
	while (current.get().type == LogicalOperatorType::LOGICAL_PROJECTION ||
	       current.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		if (current.get().children.size() != 1) {
			return nullptr;
		}
		current = *current.get().children[0];
	}
	if (current.get().type != LogicalOperatorType::LOGICAL_GET) {
		return nullptr;
	}
	auto &get = current.get().Cast<LogicalGet>();
	if (get.function.name != "seq_scan" || !get.GetTable()) {
		return nullptr;
	}
	return &get;
}

//! The number of Top-N operators whose input is being optimized on this thread (TopN::Optimize is recursive and one
//! query's optimizer runs on one thread): a Top-N found below another one feeds it, and its bound keeps the plain wrapper
static thread_local idx_t rowwise_top_n_enclosing_count = 0;

struct RowwiseTopNEnclosingScope {
	RowwiseTopNEnclosingScope() {
		rowwise_top_n_enclosing_count++;
	}
	~RowwiseTopNEnclosingScope() {
		rowwise_top_n_enclosing_count--;
	}
};

//! The order types whose bound is applied row by row: fixed-width integers, decimals, dates, times and timestamps,
//! compared by the scan's constant-comparison path on the column's own storage type
static bool RowwiseTopNBoundType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::UHUGEINT:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
		return true;
	default:
		return false;
	}
}

TopN::TopN(ClientContext &context_p) : context(context_p) {
}

bool TopN::CanOptimize(LogicalOperator &op, optional_ptr<ClientContext> context) {
	if (op.type == LogicalOperatorType::LOGICAL_LIMIT) {
		auto &limit = op.Cast<LogicalLimit>();

		if (limit.limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
			// we need LIMIT to be present AND be a constant value for us to be able to use Top-N
			return false;
		}
		if (limit.offset_val.Type() == LimitNodeType::EXPRESSION_VALUE) {
			// we need offset to be either not set (i.e. limit without offset) OR have offset be
			return false;
		}

		auto child_op = op.children[0].get();
		if (context) {
			// estimate child cardinality if the context is available
			child_op->EstimateCardinality(*context);
		}

		if (child_op->has_estimated_cardinality) {
			// only check if we should switch to full sorting if we have estimated cardinality
			auto constant_limit = static_cast<double>(limit.limit_val.GetConstantValue());
			if (limit.offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
				constant_limit += static_cast<double>(limit.offset_val.GetConstantValue());
			}
			auto child_card = static_cast<double>(child_op->estimated_cardinality);

			// if the limit is > 0.7% of the child cardinality, sorting the whole table is faster
			bool limit_is_large = constant_limit > 5000;
			if (constant_limit > child_card * 0.007 && limit_is_large) {
				return false;
			}
		}

		while (child_op->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			D_ASSERT(!child_op->children.empty());
			child_op = child_op->children[0].get();
		}

		return child_op->type == LogicalOperatorType::LOGICAL_ORDER_BY;
	}
	return false;
}

void TopN::PushdownDynamicFilters(LogicalTopN &op) {
	// pushdown dynamic filters through the Top-N operator
	bool nulls_first = op.orders[0].null_order == OrderByNullType::NULLS_FIRST;
	auto &type = op.orders[0].expression->return_type;
	if (!TypeIsIntegral(type.InternalType()) && type.id() != LogicalTypeId::VARCHAR) {
		// only supported for integral types currently
		return;
	}
	if (op.orders[0].expression->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		// we can only pushdown on ORDER BY [col] currently
		return;
	}
	if (op.dynamic_filter) {
		// dynamic filter is already set
		return;
	}
	auto &colref = op.orders[0].expression->Cast<BoundColumnRefExpression>();
	vector<JoinFilterPushdownColumn> columns;
	JoinFilterPushdownColumn column;
	column.probe_column_index = colref.binding;
	columns.emplace_back(column);
	vector<PushdownFilterTarget> pushdown_targets;
	JoinFilterPushdownOptimizer::GetPushdownFilterTargets(*op.children[0], std::move(columns), pushdown_targets);
	if (pushdown_targets.empty()) {
		// no pushdown targets
		return;
	}
	// found pushdown targets! generate dynamic filters
	ExpressionType comparison_type;
	if (op.orders[0].type == OrderType::ASCENDING) {
		// for ascending order, we want the lowest N elements, so we filter on C <= [boundary]
		// if we only have a single order clause, we can filter on C < boundary
		comparison_type =
		    op.orders.size() == 1 ? ExpressionType::COMPARE_LESSTHAN : ExpressionType::COMPARE_LESSTHANOREQUALTO;
	} else {
		// for descending order, we want the highest N elements, so we filter on C >= [boundary]
		// if we only have a single order clause, we can filter on C > boundary
		comparison_type =
		    op.orders.size() == 1 ? ExpressionType::COMPARE_GREATERTHAN : ExpressionType::COMPARE_GREATERTHANOREQUALTO;
	}
	Value minimum_value = type.InternalType() == PhysicalType::VARCHAR ? Value("") : Value::MinimumValue(type);
	auto base_filter = DynamicFilterData::CreateBound(comparison_type, std::move(minimum_value));
	auto filter_data = make_shared_ptr<DynamicFilterData>();
	filter_data->filter = std::move(base_filter);

	// put the filter into the Top-N clause
	op.dynamic_filter = filter_data;

	// the scan on which the bound is applied row by row: NULLS LAST only (a NULLS FIRST bound is an OR with IS NULL),
	// and never for a Top-N whose result feeds another Top-N
	optional_ptr<LogicalGet> rowwise_get;
	if (!nulls_first && rowwise_top_n_enclosing_count == 0 && RowwiseTopNBoundEnabled()) {
		rowwise_get = RowwiseTopNScan(*op.children[0]);
	}

	for (auto &target : pushdown_targets) {
		auto &get = target.get;
		D_ASSERT(target.columns.size() == 1);
		auto col_idx = target.columns[0].probe_column_index.column_index;

		// create the actual dynamic filter
		auto dynamic_filter = make_uniq<DynamicFilter>(filter_data);
		unique_ptr<TableFilter> pushed_filter = std::move(dynamic_filter);
		if (nulls_first) {
			auto or_filter = make_uniq<ConjunctionOrFilter>();
			or_filter->child_filters.push_back(make_uniq<IsNullFilter>());
			or_filter->child_filters.push_back(std::move(pushed_filter));
			pushed_filter = std::move(or_filter);
		}
		unique_ptr<TableFilter> optional_filter;
		if (rowwise_get && &get == rowwise_get.get() && target.columns[0].storage_type == type &&
		    RowwiseTopNBoundType(type)) {
			// applied row by row while it filters enough (the join min/max policy), zonemap-only once it does not
			optional_filter = make_uniq<SelectivityOptionalFilter>(std::move(pushed_filter),
			                                                       SelectivityOptionalFilter::MIN_MAX_THRESHOLD,
			                                                       SelectivityOptionalFilter::MIN_MAX_CHECK_N, true);
		} else {
			optional_filter = make_uniq<OptionalFilter>(std::move(pushed_filter));
		}

		// push the filter into the table scan
		auto &column_index = get.GetColumnIds()[col_idx];
		get.table_filters.PushFilter(column_index, std::move(optional_filter));
	}
}

unique_ptr<LogicalOperator> TopN::Optimize(unique_ptr<LogicalOperator> op) {
	if (CanOptimize(*op, &context)) {
		vector<unique_ptr<LogicalOperator>> projections;

		// traverse operator tree and collect all projection nodes until we reach
		// the order by operator

		auto child = std::move(op->children[0]);
		// collect all projections until we get to the order by
		while (child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			D_ASSERT(!child->children.empty());
			auto tmp = std::move(child->children[0]);
			projections.push_back(std::move(child));
			child = std::move(tmp);
		}
		D_ASSERT(child->type == LogicalOperatorType::LOGICAL_ORDER_BY);
		auto &order_by = child->Cast<LogicalOrder>();

		// Move order by operator into children of limit operator
		op->children[0] = std::move(child);

		auto &limit = op->Cast<LogicalLimit>();
		auto limit_val = limit.limit_val.GetConstantValue();
		idx_t offset_val = 0;
		if (limit.offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
			offset_val = limit.offset_val.GetConstantValue();
		}
		auto topn = make_uniq<LogicalTopN>(std::move(order_by.orders), limit_val, offset_val);
		topn->AddChild(std::move(order_by.children[0]));
		auto cardinality = limit_val;
		if (topn->children[0]->has_estimated_cardinality && topn->children[0]->estimated_cardinality < limit_val) {
			cardinality = topn->children[0]->estimated_cardinality;
		}
		topn->SetEstimatedCardinality(cardinality);
		op = std::move(topn);

		// reconstruct all projection nodes above limit operator
		while (!projections.empty()) {
			auto node = std::move(projections.back());
			node->children[0] = std::move(op);
			op = std::move(node);
			projections.pop_back();
		}
	}
	if (op->type == LogicalOperatorType::LOGICAL_TOP_N) {
		PushdownDynamicFilters(op->Cast<LogicalTopN>());
		// the operators below this Top-N feed it: a Top-N among them keeps the plain wrapper
		RowwiseTopNEnclosingScope enclosing;
		for (auto &child : op->children) {
			child = Optimize(std::move(child));
		}
		return op;
	}

	for (auto &child : op->children) {
		child = Optimize(std::move(child));
	}
	return op;
}

} // namespace duckdb
