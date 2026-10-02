#include "duckdb/execution/operator/aggregate/run_aggregate.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/operator/aggregate/fused_integer_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <cstdlib>

namespace duckdb {

namespace {

//! Rank and signedness of the run-aware integer input types
bool RunAwareIntegerRank(const LogicalType &type, idx_t &rank, bool &is_signed) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		rank = 1;
		is_signed = true;
		return true;
	case LogicalTypeId::SMALLINT:
		rank = 2;
		is_signed = true;
		return true;
	case LogicalTypeId::INTEGER:
		rank = 3;
		is_signed = true;
		return true;
	case LogicalTypeId::BIGINT:
		rank = 4;
		is_signed = true;
		return true;
	case LogicalTypeId::UTINYINT:
		rank = 1;
		is_signed = false;
		return true;
	case LogicalTypeId::USMALLINT:
		rank = 2;
		is_signed = false;
		return true;
	case LogicalTypeId::UINTEGER:
		rank = 3;
		is_signed = false;
		return true;
	case LogicalTypeId::UBIGINT:
		rank = 4;
		is_signed = false;
		return true;
	default:
		return false;
	}
}

bool IsRunAwareInputType(const LogicalType &type) {
	idx_t rank;
	bool is_signed;
	return RunAwareIntegerRank(type, rank, is_signed);
}

//! Every value of source is representable in target (so casting each run value equals casting each row)
bool IsExactWideningIntegerCast(const LogicalType &source, const LogicalType &target) {
	idx_t source_rank, target_rank;
	bool source_signed, target_signed;
	if (!RunAwareIntegerRank(source, source_rank, source_signed) ||
	    !RunAwareIntegerRank(target, target_rank, target_signed)) {
		return false;
	}
	if (source_signed && !target_signed) {
		return false;
	}
	return target_rank > source_rank;
}

//! Whether a run partial hands its fused state over at its seal; off, every hand-over waits for the aggregate's
//! Finalize
bool RunPartialSealEnabled() {
	return kRunPartialSeal;
}

//! Whether a run partial builds its fused state outside the descriptor's lock
bool RunPartialLocalEnabled() {
	return kRunPartialLocalBuild;
}

} // namespace

//===--------------------------------------------------------------------===//
// RunAggregateData
//===--------------------------------------------------------------------===//
RunAggregateData::RunAggregateData(const PhysicalUngroupedAggregate &op_p, vector<RunAggregateColumn> columns_p,
                                   vector<idx_t> nullary_p, vector<aggregate_run_update_t> run_updates_p,
                                   string column_names_p)
    : ungrouped_op(&op_p), columns(std::move(columns_p)), nullary_aggregates(std::move(nullary_p)),
      run_updates(std::move(run_updates_p)), column_names(std::move(column_names_p)) {
}

RunAggregateData::RunAggregateData(const PhysicalHashAggregate &op_p, vector<RunAggregateColumn> columns_p,
                                   vector<aggregate_grouped_run_update_t> grouped_run_updates_p, string column_names_p)
    : grouped_op(&op_p), columns(std::move(columns_p)),
      grouped_run_updates(std::move(grouped_run_updates_p)), column_names(std::move(column_names_p)) {
}

RunAggregateData::~RunAggregateData() {
}

void RunAggregateData::TryAttach(ClientContext &context, PhysicalUngroupedAggregate &op) {
	if (op.distinct_data || op.children.empty()) {
		return;
	}
	// the child chain must be PROJECTION* -> TABLE_SCAN
	vector<reference<PhysicalProjection>> projections;
	reference<PhysicalOperator> child(op.children[0]);
	while (child.get().type == PhysicalOperatorType::PROJECTION) {
		auto &projection = child.get().Cast<PhysicalProjection>();
		projections.push_back(projection);
		if (projection.children.empty()) {
			return;
		}
		child = projection.children[0];
	}
	if (child.get().type != PhysicalOperatorType::TABLE_SCAN) {
		return;
	}
	auto &scan = child.get().Cast<PhysicalTableScan>();
	if (scan.function.name != "seq_scan" || !scan.function.function || !scan.bind_data || scan.run_aggregate) {
		return;
	}
	auto &bind_data = scan.bind_data->Cast<TableScanBindData>();
	if (bind_data.is_create_index || bind_data.is_index_scan || bind_data.order_options) {
		return;
	}
	if (bind_data.table.type != CatalogType::TABLE_ENTRY || !bind_data.table.IsDuckTable()) {
		return;
	}
	if (scan.table_filters && !scan.table_filters->filters.empty()) {
		return;
	}
	if (scan.dynamic_filters || scan.extra_info.sample_options || scan.column_ids.empty()) {
		return;
	}
	vector<RunAggregateColumn> columns;
	vector<idx_t> nullary;
	vector<aggregate_run_update_t> run_updates;
	vector<bool> covered(scan.column_ids.size(), false);
	for (idx_t aggr_idx = 0; aggr_idx < op.aggregates.size(); aggr_idx++) {
		auto &aggr = op.aggregates[aggr_idx]->Cast<BoundAggregateExpression>();
		auto run_update = AggregateFunction::GetRunUpdate(aggr.function);
		if (aggr.IsDistinct() || aggr.filter || aggr.order_bys || !run_update) {
			return;
		}
		run_updates.push_back(run_update);
		if (aggr.children.empty()) {
			nullary.push_back(aggr_idx);
			continue;
		}
		if (aggr.children.size() != 1 || aggr.children[0]->GetExpressionType() != ExpressionType::BOUND_REF) {
			return;
		}
		idx_t index = aggr.children[0]->Cast<BoundReferenceExpression>().index;
		unique_ptr<Expression> cast;
		for (auto &projection_ref : projections) {
			auto &projection = projection_ref.get();
			if (index >= projection.select_list.size()) {
				return;
			}
			auto &expr = *projection.select_list[index];
			if (expr.GetExpressionType() == ExpressionType::BOUND_REF) {
				index = expr.Cast<BoundReferenceExpression>().index;
				continue;
			}
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_CAST && !cast) {
				auto &cast_expr = expr.Cast<BoundCastExpression>();
				if (cast_expr.child->GetExpressionType() != ExpressionType::BOUND_REF ||
				    !IsExactWideningIntegerCast(cast_expr.child->return_type, cast_expr.return_type)) {
					return;
				}
				cast = expr.Copy();
				index = cast_expr.child->Cast<BoundReferenceExpression>().index;
				continue;
			}
			return;
		}
		// index now addresses the scan's output; map it to the scan column position
		idx_t scan_index = index;
		if (!scan.projection_ids.empty()) {
			if (index >= scan.projection_ids.size()) {
				return;
			}
			scan_index = scan.projection_ids[index];
		}
		if (scan_index >= scan.column_ids.size()) {
			return;
		}
		auto &column_index = scan.column_ids[scan_index];
		if (column_index.HasChildren() || !column_index.HasPrimaryIndex() || column_index.IsVirtualColumn()) {
			return;
		}
		auto column_id = column_index.GetPrimaryIndex();
		if (column_id >= scan.returned_types.size() || column_id >= scan.names.size()) {
			return;
		}
		auto &storage_type = scan.returned_types[column_id];
		if (!IsRunAwareInputType(storage_type)) {
			return;
		}
		auto &produced = cast ? cast->return_type : storage_type;
		// the bound child's type is the function's concrete argument type whenever the function declares one; the unary
		// count declares ANY (the SUM(x + k) rewrite binds it over the scan column) and its run update reads no value
		if (produced != aggr.children[0]->return_type) {
			return;
		}
		if (cast && cast->Cast<BoundCastExpression>().child->return_type != storage_type) {
			return;
		}
		// one entry per scan column: the partial registers one run sink per scanned column, so aggregates reading the
		// same column through different casts keep the ordinary plan
		bool found = false;
		for (auto &column : columns) {
			if (column.scan_column_index != scan_index) {
				continue;
			}
			bool same_cast = (!column.cast && !cast) || (column.cast && cast && column.cast->Equals(*cast));
			if (!same_cast) {
				return;
			}
			column.aggregates.push_back(aggr_idx);
			found = true;
			break;
		}
		if (!found) {
			RunAggregateColumn column;
			column.scan_column_index = scan_index;
			column.storage_type = storage_type;
			column.cast = std::move(cast);
			column.aggregates.push_back(aggr_idx);
			columns.push_back(std::move(column));
		}
		covered[scan_index] = true;
	}
	if (columns.empty()) {
		return;
	}
	for (idx_t i = 0; i < covered.size(); i++) {
		if (!covered[i]) {
			// a scanned column that no run-aware aggregate consumes: keep the plan as is
			return;
		}
	}
	string column_names;
	for (auto &column : columns) {
		auto column_id = scan.column_ids[column.scan_column_index].GetPrimaryIndex();
		if (!column_names.empty()) {
			column_names += ", ";
		}
		column_names += scan.names[column_id];
		if (column.cast) {
			column.cast->Cast<BoundCastExpression>().child =
			    make_uniq<BoundReferenceExpression>(column.storage_type, 0);
			column_names += " (cast)";
		}
	}
	auto data = make_shared_ptr<RunAggregateData>(op, std::move(columns), std::move(nullary), std::move(run_updates),
	                                              std::move(column_names));
	op.run_aggregate = data;
	scan.run_aggregate = data;
}

//===--------------------------------------------------------------------===//
// RunAggregateData: the grouped form
//===--------------------------------------------------------------------===//
void RunAggregateData::TryAttachGrouped(ClientContext &context, PhysicalHashAggregate &op) {
	if (op.children.empty() || op.distinct_collection_info) {
		return;
	}
	// exactly one grouping set over exactly one group expression, no GROUPING()/ROLLUP/CUBE
	if (op.groupings.size() != 1 || op.grouping_sets.size() != 1) {
		return;
	}
	auto &grouping = op.groupings[0];
	if (grouping.distinct_data) {
		return;
	}
	auto &radix = grouping.table_data;
	if (radix.grouping_set.size() != 1 || *radix.grouping_set.begin() != 0 || !radix.null_groups.empty() ||
	    radix.group_types.size() != 1) {
		return;
	}
	auto &grouped_data = op.grouped_aggregate_data;
	if (grouped_data.groups.size() != 1 || !grouped_data.grouping_functions.empty()) {
		return;
	}
	auto &group_expr = *grouped_data.groups[0];
	if (group_expr.GetExpressionType() != ExpressionType::BOUND_REF) {
		return;
	}
	// every aggregate must be nullary (the COUNT(*) family) and carry a grouped run update: a unary aggregate would
	// read a second scan column whose runs need not align with the group column's, which the run path cannot honour
	vector<aggregate_grouped_run_update_t> grouped_run_updates;
	for (auto &aggregate : grouped_data.aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (aggr.IsDistinct() || aggr.filter || aggr.order_bys || !aggr.children.empty()) {
			return;
		}
		auto run_update = AggregateFunction::GetGroupedRunUpdate(aggr.function);
		if (!run_update) {
			return;
		}
		grouped_run_updates.push_back(run_update);
	}
	if (grouped_run_updates.empty()) {
		return;
	}
	// the child chain must be PROJECTION* -> TABLE_SCAN, exactly as the ungrouped attach requires
	vector<reference<PhysicalProjection>> projections;
	reference<PhysicalOperator> child(op.children[0]);
	while (child.get().type == PhysicalOperatorType::PROJECTION) {
		auto &projection = child.get().Cast<PhysicalProjection>();
		projections.push_back(projection);
		if (projection.children.empty()) {
			return;
		}
		child = projection.children[0];
	}
	if (child.get().type != PhysicalOperatorType::TABLE_SCAN) {
		return;
	}
	auto &scan = child.get().Cast<PhysicalTableScan>();
	if (scan.function.name != "seq_scan" || !scan.function.function || !scan.bind_data || scan.run_aggregate) {
		return;
	}
	auto &bind_data = scan.bind_data->Cast<TableScanBindData>();
	if (bind_data.is_create_index || bind_data.is_index_scan || bind_data.order_options) {
		return;
	}
	if (bind_data.table.type != CatalogType::TABLE_ENTRY || !bind_data.table.IsDuckTable()) {
		return;
	}
	if (scan.table_filters && !scan.table_filters->filters.empty()) {
		return;
	}
	if (scan.dynamic_filters || scan.extra_info.sample_options) {
		return;
	}
	// the scan must project exactly the group column: the run branch requires every projected column to be
	// run-eligible, and only a single column keeps a run's group key constant over the whole run
	if (scan.column_ids.size() != 1) {
		return;
	}
	// resolve the group reference through the projections down to the scan's output
	idx_t index = group_expr.Cast<BoundReferenceExpression>().index;
	for (auto &projection_ref : projections) {
		auto &projection = projection_ref.get();
		if (index >= projection.select_list.size()) {
			return;
		}
		auto &expr = *projection.select_list[index];
		if (expr.GetExpressionType() != ExpressionType::BOUND_REF) {
			return;
		}
		index = expr.Cast<BoundReferenceExpression>().index;
	}
	idx_t scan_index = index;
	if (!scan.projection_ids.empty()) {
		if (index >= scan.projection_ids.size()) {
			return;
		}
		scan_index = scan.projection_ids[index];
	}
	if (scan_index != 0) {
		return;
	}
	auto &column_index = scan.column_ids[scan_index];
	if (column_index.HasChildren() || !column_index.HasPrimaryIndex() || column_index.IsVirtualColumn()) {
		return;
	}
	auto column_id = column_index.GetPrimaryIndex();
	if (column_id >= scan.returned_types.size() || column_id >= scan.names.size()) {
		return;
	}
	auto &storage_type = scan.returned_types[column_id];
	if (!IsRunAwareInputType(storage_type)) {
		return;
	}
	// the hash table stores the group in its own type: no cast may sit between the scan column and the group
	if (storage_type != group_expr.return_type || storage_type != radix.group_types[0]) {
		return;
	}
	RunAggregateColumn column;
	column.scan_column_index = scan_index;
	column.storage_type = storage_type;
	vector<RunAggregateColumn> columns;
	columns.push_back(std::move(column));
	auto data = make_shared_ptr<RunAggregateData>(op, std::move(columns), std::move(grouped_run_updates),
	                                              scan.names[column_id]);
	op.run_aggregate = data;
	scan.run_aggregate = data;
}

void RunAggregateData::ResetGrouped(GlobalSinkState &radix_sink, optional_ptr<FusedAggregateGlobalState> fused_sink_p) {
	lock_guard<mutex> guard(lock);
	grouped_sink = &radix_sink;
	grouped_partials.clear();
	fused_sink = fused_sink_p;
	fused_partials.clear();
}

LocalSinkState &RunAggregateData::CreateGroupedLocal(ClientContext &context) {
	auto state = grouped_op->groupings[0].table_data.GetLocalSinkState(context);
	lock_guard<mutex> guard(lock);
	grouped_partials.push_back(std::move(state));
	return *grouped_partials.back();
}

void RunAggregateData::SinkGroupedRuns(ClientContext &context, DataChunk &groups, const uint16_t *counts,
                                       idx_t run_count, LocalSinkState &lstate) {
	if (!grouped_sink) {
		throw InternalException("run-aware grouped aggregate: the radix sink state is not registered");
	}
	grouped_op->groupings[0].table_data.SinkRuns(context, *grouped_sink, lstate, groups, counts, run_count,
	                                             grouped_run_updates);
}

vector<unique_ptr<LocalSinkState>> RunAggregateData::TakeGroupedPartials() {
	lock_guard<mutex> guard(lock);
	return std::move(grouped_partials);
}

optional_ptr<FusedAggregateLocalState> RunAggregateData::CreateFusedLocal(LocalSinkState &radix_local) {
	if (RunPartialLocalEnabled()) {
		{
			lock_guard<mutex> guard(lock);
			if (!fused_sink) {
				return nullptr;
			}
		}
		// the thread's fused state (its 4096-entry arrays and line buffer) is built before the lock is taken, so
		// the scan threads build theirs in parallel; the lock only registers it
		auto state = make_uniq<FusedAggregateLocalState>(*grouped_op->fused);
		state->run_radix_local = &radix_local;
		lock_guard<mutex> guard(lock);
		fused_partials.push_back(std::move(state));
		return fused_partials.back().get();
	}
	lock_guard<mutex> guard(lock);
	if (!fused_sink) {
		return nullptr;
	}
	auto state = make_uniq<FusedAggregateLocalState>(*grouped_op->fused);
	state->run_radix_local = &radix_local;
	fused_partials.push_back(std::move(state));
	return fused_partials.back().get();
}

bool RunAggregateData::SinkFusedRuns(ClientContext &context, Vector &values, const uint16_t *counts, idx_t run_count,
                                     FusedAggregateLocalState &lstate) {
	return grouped_op->fused->SinkRuns(context, values, counts, run_count, *grouped_op, *fused_sink, lstate);
}

void RunAggregateData::CombineFusedPartials(ClientContext &context) {
	vector<unique_ptr<FusedAggregateLocalState>> partials;
	{
		lock_guard<mutex> guard(lock);
		partials = std::move(fused_partials);
		fused_partials.clear();
	}
	for (auto &partial : partials) {
		grouped_op->fused->CombineRuns(context, *grouped_op, *fused_sink, *partial);
	}
}

void RunAggregateData::Reset(Allocator &client_allocator) {
	lock_guard<mutex> guard(lock);
	allocator = make_uniq<ArenaAllocator>(client_allocator);
	shared = make_uniq<UngroupedAggregateState>(ungrouped_op->aggregates);
	rows = 0;
	merged = false;
}

void RunAggregateData::CombineInto(GlobalUngroupedAggregateState &gstate) {
	lock_guard<mutex> guard(lock);
	if (merged || !shared) {
		return;
	}
	auto &aggregates = ungrouped_op->aggregates;
	for (idx_t i = 0; i < aggregates.size(); i++) {
		auto &aggr = aggregates[i]->Cast<BoundAggregateExpression>();
		Vector source(Value::POINTER(CastPointerToValue(shared->aggregate_data[i].get())));
		Vector target(Value::POINTER(CastPointerToValue(gstate.state.aggregate_data[i].get())));
		AggregateInputData input(aggr.bind_info.get(), gstate.allocator, AggregateCombineType::ALLOW_DESTRUCTIVE);
		aggr.function.combine(source, target, input, 1);
#ifdef DEBUG
		gstate.state.counts[i] += rows;
#endif
	}
	merged = true;
}

unique_ptr<RunAggregatePartial> RunAggregateData::CreatePartial(ClientContext &context) {
	return make_uniq<RunAggregatePartial>(*this, context);
}

//===--------------------------------------------------------------------===//
// RunAggregatePartial
//===--------------------------------------------------------------------===//
struct RunAggregatePartial::ColumnSink : public RunSink {
	ColumnSink(RunAggregatePartial &partial_p, const RunAggregateColumn &column_p, ClientContext &context)
	    : partial(partial_p), column(column_p), value_vector(column_p.storage_type, RunAggregateData::RUN_BATCH_MAX_RUNS),
	      count_buffer(make_unsafe_uniq_array_uninitialized<uint16_t>(RunAggregateData::RUN_BATCH_MAX_RUNS)),
	      cast_output(column_p.cast ? column_p.cast->return_type : column_p.storage_type,
	                  RunAggregateData::RUN_BATCH_MAX_RUNS) {
		values = FlatVector::GetData(value_vector);
		counts = count_buffer.get();
		capacity = RunAggregateData::RUN_BATCH_MAX_RUNS;
		if (column.cast) {
			cast_executor = make_uniq<ExpressionExecutor>(context, *column.cast);
			vector<LogicalType> types;
			types.push_back(column.storage_type);
			cast_input.InitializeEmpty(types);
		}
	}

	void Flush() override {
		if (run_count == 0) {
			return;
		}
		if (partial.data.IsGrouped()) {
			partial.ApplyGroupedRuns(value_vector, count_buffer.get(), run_count);
			run_count = 0;
			return;
		}
		if (cast_executor) {
			cast_input.data[0].Reference(value_vector);
			cast_input.SetCardinality(run_count);
			cast_executor->ExecuteExpression(cast_input, cast_output);
			if (cast_output.GetVectorType() != VectorType::FLAT_VECTOR) {
				cast_output.Flatten(run_count);
			}
			partial.ApplyRuns(cast_output, count_buffer.get(), run_count, column.aggregates);
		} else {
			partial.ApplyRuns(value_vector, count_buffer.get(), run_count, column.aggregates);
		}
		run_count = 0;
	}

	RunAggregatePartial &partial;
	const RunAggregateColumn &column;
	Vector value_vector;
	unsafe_unique_array<uint16_t> count_buffer;
	unique_ptr<ExpressionExecutor> cast_executor;
	DataChunk cast_input;
	Vector cast_output;
};

RunAggregatePartial::RunAggregatePartial(RunAggregateData &data_p, ClientContext &context_p)
    : data(data_p), allocator(Allocator::Get(context_p)), nullary_input(LogicalType::BIGINT, 1), context(context_p) {
	static_assert(STANDARD_VECTOR_SIZE <= 65535, "run lengths are stored as uint16_t");
	if (data.IsGrouped()) {
		// one thread-local radix sink state, owned by the shared descriptor so it outlives this scan-local partial
		grouped_local = &data.CreateGroupedLocal(context_p);
		fused_local = data.CreateFusedLocal(*grouped_local);
		vector<LogicalType> group_types;
		group_types.push_back(data.columns[0].storage_type);
		group_chunk.InitializeEmpty(group_types);
	} else {
		local = make_uniq<UngroupedAggregateState>(data.ungrouped_op->aggregates);
	}
	idx_t max_index = 0;
	for (auto &column : data.columns) {
		max_index = MaxValue<idx_t>(max_index, column.scan_column_index + 1);
	}
	by_scan_column.resize(max_index, nullptr);
	for (auto &column : data.columns) {
		sinks.push_back(make_uniq<ColumnSink>(*this, column, context_p));
		by_scan_column[column.scan_column_index] = sinks.back().get();
	}
}

RunAggregatePartial::~RunAggregatePartial() {
}

RunSink &RunAggregatePartial::GetSink(idx_t scan_column_index) {
	if (scan_column_index >= by_scan_column.size() || !by_scan_column[scan_column_index]) {
		throw InternalException("run-aware aggregate: no run sink for scan column %llu", scan_column_index);
	}
	return *by_scan_column[scan_column_index];
}

void RunAggregatePartial::ApplyRuns(Vector &values, const uint16_t *counts, idx_t run_count,
                                    const vector<idx_t> &aggregates) {
	auto &exprs = data.ungrouped_op->aggregates;
	for (auto aggr_idx : aggregates) {
		auto &aggr = exprs[aggr_idx]->Cast<BoundAggregateExpression>();
		AggregateInputData input(aggr.bind_info.get(), allocator);
		data.run_updates[aggr_idx](values, counts, run_count, input, local->aggregate_data[aggr_idx].get());
	}
}

void RunAggregatePartial::ApplyGroupedRuns(Vector &values, const uint16_t *counts, idx_t run_count) {
	if (fused_local && data.SinkFusedRuns(context, values, counts, run_count, *fused_local)) {
		return;
	}
	group_chunk.data[0].Reference(values);
	group_chunk.SetCardinality(run_count);
	data.SinkGroupedRuns(context, group_chunk, counts, run_count, *grouped_local);
}

void RunAggregatePartial::AddRows(idx_t count) {
	rows += count;
	if (data.IsGrouped() || data.nullary_aggregates.empty()) {
		return;
	}
	// a vector holds at most STANDARD_VECTOR_SIZE rows, which fits the run length type
	uint16_t run = UnsafeNumericCast<uint16_t>(count);
	auto &exprs = data.ungrouped_op->aggregates;
	for (auto aggr_idx : data.nullary_aggregates) {
		auto &aggr = exprs[aggr_idx]->Cast<BoundAggregateExpression>();
		AggregateInputData input(aggr.bind_info.get(), allocator);
		data.run_updates[aggr_idx](nullary_input, &run, 1, input, local->aggregate_data[aggr_idx].get());
	}
}

void RunAggregatePartial::CombineIntoShared() {
	for (auto &sink : sinks) {
		sink->Flush();
	}
	if (data.IsGrouped()) {
		// the grouped runs are already in this thread's hash table; it is combined at the aggregate's Finalize
		return;
	}
	if (rows == 0) {
		return;
	}
	auto &exprs = data.ungrouped_op->aggregates;
	lock_guard<mutex> guard(data.lock);
	if (!data.shared) {
		throw InternalException("run-aware aggregate: shared states are not initialized");
	}
	for (idx_t i = 0; i < exprs.size(); i++) {
		auto &aggr = exprs[i]->Cast<BoundAggregateExpression>();
		Vector source(Value::POINTER(CastPointerToValue(local->aggregate_data[i].get())));
		Vector target(Value::POINTER(CastPointerToValue(data.shared->aggregate_data[i].get())));
		AggregateInputData input(aggr.bind_info.get(), *data.allocator, AggregateCombineType::ALLOW_DESTRUCTIVE);
		aggr.function.combine(source, target, input, 1);
		aggr.function.initialize(aggr.function, local->aggregate_data[i].get());
	}
	data.rows += rows;
	rows = 0;
}

void RunAggregatePartial::FinishScan() {
	if (!fused_local || !RunPartialSealEnabled()) {
		return;
	}
	// the scan that owns this partial has sunk its last run (each exhausted row group flushed its batches), so its
	// fused state leaves the parked set under the descriptor's lock - exactly one of this call and the aggregate's
	// Finalize takes it - and is handed over (or drained into grouped_local when the kernel is abandoned) here, on this
	// scan's thread, in parallel with the other scans: the same CombineRuns call Finalize would make, on the same state
	unique_ptr<FusedAggregateLocalState> state;
	{
		lock_guard<mutex> guard(data.lock);
		auto &partials = data.fused_partials;
		for (auto it = partials.begin(); it != partials.end(); ++it) {
			if (it->get() == fused_local.get()) {
				state = std::move(*it);
				partials.erase(it);
				break;
			}
		}
	}
	// no run may follow: one would find no fused state and no radix state and throw instead of being lost
	fused_local = nullptr;
	grouped_local = nullptr;
	if (state) {
		data.grouped_op->fused->CombineRuns(context, *data.grouped_op, *data.fused_sink, *state);
	}
}

} // namespace duckdb
