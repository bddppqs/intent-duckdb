
//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/filter/optional_filter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/table_filter.hpp"
#include "duckdb/planner/table_filter_state.hpp"

namespace duckdb {

class OptionalFilter : public TableFilter {
public:
	static constexpr auto TYPE = TableFilterType::OPTIONAL_FILTER;

public:
	explicit OptionalFilter(unique_ptr<TableFilter> filter = nullptr);

	//! Optional child filters.
	unique_ptr<TableFilter> child_filter;

public:
	string ToString(const string &column_name) const override;
	unique_ptr<TableFilter> Copy() const override;
	unique_ptr<Expression> ToExpression(const Expression &column) const override;
	FilterPropagateResult CheckStatistics(BaseStatistics &stats) const override;
	void Serialize(Serializer &serializer) const override;
	static unique_ptr<TableFilter> Deserialize(Deserializer &deserializer);

	virtual void FiltersNullValues(const LogicalType &type, bool &filters_nulls, bool &filters_valid_values,
	                               TableFilterState &filter_state) const {
	}

	virtual unique_ptr<TableFilterState> InitializeState(ClientContext &context) const {
		return make_uniq<TableFilterState>();
	}

	//! Whether the scan keeps this optional filter in its per-vector loop for the current row group (it is otherwise
	//! used for zonemap checks only). False for every optional filter except a SelectivityOptionalFilter wrapping a
	//! Top-N bound whose thread-local state is still active.
	virtual bool FiltersRows(TableFilterState &filter_state) const {
		return false;
	}

	virtual idx_t FilterSelection(SelectionVector &sel, Vector &vector, UnifiedVectorFormat &vdata,
	                              TableFilterState &filter_state, idx_t scan_count, idx_t &approved_tuple_count) const;
};

} // namespace duckdb
