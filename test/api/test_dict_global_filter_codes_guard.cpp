#include "catch.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/storage/compression/dict_global/persisted_translation.hpp"

using namespace duckdb;
using namespace std;

// A codes-only column decides its pushed filter by code class (NULL, the empty string, any other), reading any constant
// comparison as the empty-string test: a filter that is not decided on codes must be refused, never mis-selected
TEST_CASE("dict_global: the code-class verdicts refuse a filter that is not decided on codes", "[dict_global]") {
	const uint32_t empty_code = 3;
	bool null_passes = false;
	bool empty_passes = false;
	bool other_passes = false;

	ConstantFilter not_empty(ExpressionType::COMPARE_NOTEQUAL, Value(""));
	REQUIRE(dict_global::CodeTranslatable(not_empty));
	REQUIRE_NOTHROW(dict_global::CodeClassVerdicts(not_empty, empty_code, null_passes, empty_passes, other_passes));
	REQUIRE(!null_passes);
	REQUIRE(!empty_passes);
	REQUIRE(other_passes);

	ConstantFilter equals_value(ExpressionType::COMPARE_EQUAL, Value("abc"));
	REQUIRE(!dict_global::CodeTranslatable(equals_value));
	REQUIRE_THROWS_AS(dict_global::CodeClassVerdicts(equals_value, empty_code, null_passes, empty_passes, other_passes),
	                  InternalException);
}
