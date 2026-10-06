#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"

#include "duckdb/common/tuning_defaults.hpp"

#include <typeinfo>

namespace duckdb {

namespace {

//! kTopNBoundLockFree: the Top-N bound publishes immutable copies its readers load without the lock; off, the plain
//! bound read under its lock
bool TopNBoundLockFree() {
	return kTopNBoundLockFree;
}

//! A Top-N bound that also publishes an immutable copy of every value it is set to. The bound itself (the base
//! ConstantFilter) is still updated under DynamicFilterData::lock for the readers that take it. `published`, `current`
//! and `previous` are written only under that lock; a reader loads `published` (acquire) and reads the copy it points
//! to, which is never modified after its release-store and is freed no earlier than the second Reset after it was made.
class PublishingBound final : public ConstantFilter {
public:
	PublishingBound(ExpressionType comparison_type, Value constant)
	    : ConstantFilter(comparison_type, std::move(constant)) {
	}

	//! The copy of the value set last; nullptr while the bound is unset
	atomic<const ConstantFilter *> published = {nullptr};
	//! The copies made since the last Reset (this execution) and those of the execution before it
	vector<unique_ptr<ConstantFilter>> current;
	vector<unique_ptr<ConstantFilter>> previous;
};

PublishingBound *AsPublishingBound(ConstantFilter *filter) {
	if (!filter || typeid(*filter) != typeid(PublishingBound)) {
		return nullptr;
	}
	return static_cast<PublishingBound *>(filter);
}

} // namespace

unique_ptr<ConstantFilter> DynamicFilterData::CreateBound(ExpressionType comparison_type, Value constant) {
	if (!TopNBoundLockFree()) {
		return make_uniq<ConstantFilter>(comparison_type, std::move(constant));
	}
	return make_uniq<PublishingBound>(comparison_type, std::move(constant));
}

bool DynamicFilterData::LoadPublished(const ConstantFilter *&bound) const {
	// `filter` is assigned once, when the plan is built (TopN::PushdownDynamicFilters), and never reassigned
	auto publishing = AsPublishingBound(filter.get());
	if (!publishing) {
		return false;
	}
	bound = publishing->published.load(std::memory_order_acquire);
	return true;
}

DynamicFilter::DynamicFilter() : TableFilter(TableFilterType::DYNAMIC_FILTER) {
}

DynamicFilter::DynamicFilter(shared_ptr<DynamicFilterData> filter_data_p)
    : TableFilter(TableFilterType::DYNAMIC_FILTER), filter_data(std::move(filter_data_p)) {
}

FilterPropagateResult DynamicFilter::CheckStatistics(BaseStatistics &stats) const {
	if (!filter_data) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	const ConstantFilter *published;
	if (filter_data->LoadPublished(published)) {
		return published ? published->CheckStatistics(stats) : FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	lock_guard<mutex> l(filter_data->lock);
	if (!filter_data->initialized) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	return filter_data->filter->CheckStatistics(stats);
}

string DynamicFilter::ToString(const string &column_name) const {
	if (filter_data) {
		return "Dynamic Filter (" + column_name + ")";
	} else {
		return "Empty Dynamic Filter (" + column_name + ")";
	}
}

unique_ptr<Expression> DynamicFilter::ToExpression(const Expression &column) const {
	if (!filter_data || !filter_data->initialized) {
		auto bound_constant = make_uniq<BoundConstantExpression>(Value(true));
		return std::move(bound_constant);
	}
	lock_guard<mutex> l(filter_data->lock);
	return filter_data->filter->ToExpression(column);
}

bool DynamicFilter::Equals(const TableFilter &other_p) const {
	if (!TableFilter::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<DynamicFilter>();
	return other.filter_data.get() == filter_data.get();
}

unique_ptr<TableFilter> DynamicFilter::Copy() const {
	return make_uniq<DynamicFilter>(filter_data);
}

void DynamicFilterData::SetValue(Value val) {
	if (val.IsNull()) {
		return;
	}
	lock_guard<mutex> l(lock);
	filter->Cast<ConstantFilter>().constant = std::move(val);
	auto publishing = AsPublishingBound(filter.get());
	if (publishing) {
		// the copy is complete before the release-store that publishes it
		publishing->current.push_back(make_uniq<ConstantFilter>(publishing->comparison_type, publishing->constant));
		publishing->published.store(publishing->current.back().get(), std::memory_order_release);
	}
	initialized = true;
}

void DynamicFilterData::Reset() {
	lock_guard<mutex> l(lock);
	initialized = false;
	auto publishing = AsPublishingBound(filter.get());
	if (publishing) {
		// Reset runs once per execution, before its Top-N sinks (PhysicalTopN::GetGlobalSinkState). A reader of this
		// execution that loaded `published` before this store can hold only a copy of the execution before, so those copies
		// are kept one more execution; the copies of the execution before that have no reader left and are freed.
		publishing->published.store(nullptr, std::memory_order_release);
		publishing->previous = std::move(publishing->current);
		publishing->current.clear();
	}
}

} // namespace duckdb
