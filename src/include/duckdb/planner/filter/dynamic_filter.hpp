
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/filter/dynamic_filter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/common/atomic.hpp"

namespace duckdb {

struct DynamicFilterData {
	mutex lock;
	unique_ptr<ConstantFilter> filter;
	atomic<bool> initialized = {false};

	void SetValue(Value val);
	void Reset();

	//! The bound a Top-N pushes into its scans. With kTopNBoundLockFree (a build-time flag) it also publishes an
	//! immutable copy of every value it is set to, so its per-vector readers need not take `lock` (LoadPublished); the
	//! copies live in the bound object, so this struct's layout is unchanged
	static unique_ptr<ConstantFilter> CreateBound(ExpressionType comparison_type, Value constant);
	//! False when the bound does not publish its values (not made by CreateBound, or the flag is off): the caller takes
	//! `lock`. True otherwise, with `bound` the immutable copy of the value set last, or nullptr while the bound is unset
	bool LoadPublished(const ConstantFilter *&bound) const;
};

class DynamicFilter : public TableFilter {
public:
	static constexpr const TableFilterType TYPE = TableFilterType::DYNAMIC_FILTER;

public:
	DynamicFilter();
	explicit DynamicFilter(shared_ptr<DynamicFilterData> filter_data);

	//! The shared, dynamic filter data
	shared_ptr<DynamicFilterData> filter_data;

public:
	FilterPropagateResult CheckStatistics(BaseStatistics &stats) const override;
	string ToString(const string &column_name) const override;
	bool Equals(const TableFilter &other) const override;
	unique_ptr<TableFilter> Copy() const override;
	unique_ptr<Expression> ToExpression(const Expression &column) const override;
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<TableFilter> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
