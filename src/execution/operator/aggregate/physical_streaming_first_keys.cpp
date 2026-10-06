#include "duckdb/execution/operator/aggregate/physical_streaming_first_keys.hpp"

#include "duckdb/common/mutex.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value_map.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/storage/buffer_manager.hpp"

namespace duckdb {

PhysicalStreamingFirstKeys::PhysicalStreamingFirstKeys(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                       vector<unique_ptr<Expression>> groups_p,
                                                       unique_ptr<FirstKeysAggregateInfo> info,
                                                       idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::STREAMING_FIRST_KEYS, std::move(types),
                       estimated_cardinality),
      groups(std::move(groups_p)), k(info->k), probe_filters(std::move(info->probe_filters)),
      probe_column_indexes(std::move(info->probe_column_indexes)),
      probe_storage_types(std::move(info->probe_storage_types)) {
	D_ASSERT(!groups.empty());
	D_ASSERT(probe_column_indexes.size() == groups.size());
	D_ASSERT(probe_storage_types.size() == groups.size());
}

//===--------------------------------------------------------------------===//
// States
//===--------------------------------------------------------------------===//
class FirstKeysGlobalState : public GlobalOperatorState {
public:
	mutex lock;
	//! Whether the key filters have been pushed for this execution
	bool pushed = false;
};

class FirstKeysOperatorState : public OperatorState {
public:
	FirstKeysOperatorState(ExecutionContext &context, const PhysicalStreamingFirstKeys &op)
	    : executor(context.client, op.groups), addresses(LogicalType::POINTER), new_groups(STANDARD_VECTOR_SIZE),
	      emitted(0), finished(false) {
		vector<LogicalType> group_types;
		for (auto &group : op.groups) {
			group_types.push_back(group->return_type);
		}
		auto &allocator = BufferAllocator::Get(context.client);
		keys.Initialize(allocator, group_types);
		ht = make_uniq<GroupedAggregateHashTable>(context.client, allocator, group_types);
		values.resize(group_types.size());
	}

	ExpressionExecutor executor;
	//! The key columns of the current input chunk
	DataChunk keys;
	//! Distinct keys seen so far (exact GROUP BY key semantics: NULL keys form one group)
	unique_ptr<GroupedAggregateHashTable> ht;
	Vector addresses;
	SelectionVector new_groups;
	//! Keys emitted so far
	idx_t emitted;
	//! Whether k keys have been emitted (or the input ended)
	bool finished;
	//! The emitted key values per key column
	vector<vector<Value>> values;
};

unique_ptr<OperatorState> PhysicalStreamingFirstKeys::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<FirstKeysOperatorState>(context, *this);
}

unique_ptr<GlobalOperatorState> PhysicalStreamingFirstKeys::GetGlobalOperatorState(ClientContext &context) const {
	// a new execution of the same plan (PREPARE/EXECUTE) starts without the previous execution's filters
	if (probe_filters) {
		probe_filters->ClearFilters(*this);
	}
	return make_uniq<FirstKeysGlobalState>();
}

//===--------------------------------------------------------------------===//
// Key filters
//===--------------------------------------------------------------------===//
void PhysicalStreamingFirstKeys::PushKeyFilters(FirstKeysGlobalState &gstate, FirstKeysOperatorState &state) const {
	lock_guard<mutex> guard(gstate.lock);
	if (gstate.pushed) {
		return;
	}
	gstate.pushed = true;
	if (!probe_filters) {
		return;
	}
	vector<pair<idx_t, unique_ptr<TableFilter>>> filters;
	for (idx_t col = 0; col < probe_column_indexes.size(); col++) {
		if (probe_column_indexes[col] == DConstants::INVALID_INDEX) {
			// a code group (FirstKeysAggregate): its keys are codes, never pushed to the scan as a value filter
			continue;
		}
		auto &storage_type = probe_storage_types[col];
		value_set_t seen;
		vector<Value> distinct_values;
		bool has_null = false;
		for (auto &value : state.values[col]) {
			if (value.IsNull()) {
				has_null = true;
				continue;
			}
			Value cast_value = value;
			if (!cast_value.DefaultTryCastAs(storage_type)) {
				// cannot express the key in the storage type: push nothing (the SEMI join keeps the result exact)
				return;
			}
			if (seen.insert(cast_value).second) {
				distinct_values.push_back(std::move(cast_value));
			}
		}
		unique_ptr<TableFilter> filter;
		if (distinct_values.empty() && !has_null) {
			continue;
		} else if (distinct_values.size() == 1 && !has_null) {
			filter = make_uniq<ConstantFilter>(ExpressionType::COMPARE_EQUAL, std::move(distinct_values[0]));
		} else if (distinct_values.empty()) {
			filter = make_uniq<IsNullFilter>();
		} else {
			auto or_filter = make_uniq<ConjunctionOrFilter>();
			for (auto &value : distinct_values) {
				or_filter->child_filters.push_back(
				    make_uniq<ConstantFilter>(ExpressionType::COMPARE_EQUAL, std::move(value)));
			}
			if (has_null) {
				or_filter->child_filters.push_back(make_uniq<IsNullFilter>());
			}
			filter = std::move(or_filter);
		}
		filters.emplace_back(probe_column_indexes[col], std::move(filter));
	}
	for (auto &entry : filters) {
		probe_filters->PushFilter(*this, entry.first, std::move(entry.second));
	}
}

//===--------------------------------------------------------------------===//
// Execute
//===--------------------------------------------------------------------===//
OperatorResultType PhysicalStreamingFirstKeys::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                       GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &state = state_p.Cast<FirstKeysOperatorState>();
	auto &gstate = gstate_p.Cast<FirstKeysGlobalState>();
	if (state.finished) {
		chunk.SetCardinality(0);
		return OperatorResultType::FINISHED;
	}
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}
	state.keys.Reset();
	state.executor.Execute(input, state.keys);
	auto new_count = state.ht->FindOrCreateGroups(state.keys, state.addresses, state.new_groups);
	if (new_count == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}
	auto take = MinValue<idx_t>(new_count, k - state.emitted);
	chunk.Slice(state.keys, state.new_groups, take);
	for (idx_t col = 0; col < state.values.size(); col++) {
		for (idx_t i = 0; i < take; i++) {
			state.values[col].push_back(chunk.GetValue(col, i));
		}
	}
	state.emitted += take;
	if (state.emitted >= k) {
		state.finished = true;
		PushKeyFilters(gstate, state);
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorFinalizeResultType PhysicalStreamingFirstKeys::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                                    GlobalOperatorState &gstate_p,
                                                                    OperatorState &state_p) const {
	auto &state = state_p.Cast<FirstKeysOperatorState>();
	auto &gstate = gstate_p.Cast<FirstKeysGlobalState>();
	if (!state.finished) {
		// the input ended with fewer than k keys: every key seen is the key set
		state.finished = true;
		PushKeyFilters(gstate, state);
	}
	chunk.SetCardinality(0);
	return OperatorFinalizeResultType::FINISHED;
}

InsertionOrderPreservingMap<string> PhysicalStreamingFirstKeys::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	string groups_info;
	for (idx_t i = 0; i < groups.size(); i++) {
		if (i > 0) {
			groups_info += "\n";
		}
		groups_info += groups[i]->GetName();
	}
	result["Groups"] = groups_info;
	result["Keys"] = StringUtil::Format("%llu", k);
	string columns;
	for (idx_t i = 0; i < probe_column_indexes.size(); i++) {
		if (i > 0) {
			columns += ", ";
		}
		columns += probe_column_indexes[i] == DConstants::INVALID_INDEX
		               ? string("-")
		               : StringUtil::Format("%llu", probe_column_indexes[i]);
	}
	result["Probe Columns"] = columns;
	result["Key Filters"] = probe_filters ? "pushed" : "none";
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
