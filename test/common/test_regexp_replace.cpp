#include "catch.hpp"
#include "duckdb.hpp"
#include "re2/re2.h"
#include "test_helpers.hpp"

using namespace duckdb;
using duckdb_re2::RE2;

namespace {
//! xorshift64: a fixed sequence, so the generated strings are the same on every run
struct ReplaceStringGenerator {
	uint64_t state;
	uint64_t Next() {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		return state;
	}
};

struct ReplaceCase {
	string pattern;
	string rewrite;
	string options;
};
} // namespace

TEST_CASE("regexp_replace agrees with RE2::Replace on generated strings", "[regexp]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the pieces the strings are made of: ASCII letters, digits and punctuation the patterns name, newlines and tabs,
	// backslashes, and UTF-8 sequences of two, three and four bytes
	const duckdb::vector<string> pieces {"a",  "b",  "x", "z", "A",  "Z",  "0",  "7",  "/",       ":",   ".",   "=",  "@", " ",
	                             "-",  "_",  "\n", "\t", "\\", "é", "я",  "中", "\xF0\x9F\x98\x80", "ab", "www.", "key"};
	ReplaceStringGenerator generator {0x9E3779B97F4A7C15ULL};
	duckdb::vector<string> inputs {""};
	for (idx_t i = 0; i < 4000; i++) {
		string s;
		const idx_t length = generator.Next() % 24;
		for (idx_t k = 0; k < length; k++) {
			s += pieces[generator.Next() % pieces.size()];
		}
		inputs.push_back(s);
	}
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE inputs (id INTEGER, s VARCHAR)"));
	{
		Appender appender(con, "inputs");
		for (idx_t i = 0; i < inputs.size(); i++) {
			appender.AppendRow(Value::INTEGER(NumericCast<int32_t>(i)), Value(inputs[i]));
		}
	}
	const duckdb::vector<ReplaceCase> cases {
	    // anchored captures, repeats, character classes and UTF-8
	    {"^([a-z]+)=.*$", "\\1", ""},
	    {"^([a-z]+)=.*$", "<\\1>", "s"},
	    {"^(\\d+)-[a-z]*$", "\\1", "i"},
	    {"(\\w+)@[^ ]*$", "[\\1]", ""},
	    {"^(.*?)\\s*$", "\\1", ""},
	    {"^(a+)b*$", "\\1|\\0", ""},
	    {"^(\\w*)[а-я]*$", "\\1", ""},
	    {"(?s)^(.)(.).*$", "\\2\\1", ""},
	    {"^([^/]*)/.*\\z", "\\1", ""},
	    {"(x|z).*?$", "#\\1#", ""},
	    {"^(é*)[^\\n]*$", "\\1", ""},
	    {"(:)[^\xF0\x9F\x98\x80]*$", "\\\\\\1", ""},
	    {"([0-9])z*$", "<\\1>", ""},
	    {"(\\.)(?:ab)?[\\x{80}-\\x{7FF}]*$", "\\1", ""},
	    // group swaps, (?m), case folding and invalid rewrites
	    {"^(a+)(b*)$", "\\2\\1", ""},
	    {"(?m)^(a).*$", "<\\1>", ""},
	    {"([a-z]+)$", "<\\1>", ""},
	    {"(.*)$", "[\\1]", ""},
	    {"(\\w)(\\w)", "\\2\\1", ""},
	    {"a", "\\x", ""},
	    {"^(a).*$", "\\3", ""},
	    {"^(a)x*$", "\\1", "i"},
	};
	for (auto &c : cases) {
		RE2::Options options;
		options.set_log_errors(false);
		for (auto option : c.options) {
			if (option == 'i') {
				options.set_case_sensitive(false);
			} else if (option == 's') {
				options.set_dot_nl(true);
			}
		}
		RE2 pattern(c.pattern, options);
		REQUIRE(pattern.ok());
		string sql = "SELECT regexp_replace(s, " + Value(c.pattern).ToSQLString() + ", " + Value(c.rewrite).ToSQLString() +
		             (c.options.empty() ? string() : ", '" + c.options + "'") + ") FROM inputs ORDER BY id";
		auto result = con.Query(sql);
		REQUIRE_NO_FAIL(*result);
		REQUIRE(result->RowCount() == inputs.size());
		idx_t mismatches = 0;
		for (idx_t i = 0; i < inputs.size(); i++) {
			string expected = inputs[i];
			RE2::Replace(&expected, pattern, c.rewrite);
			if (result->GetValue(0, i).ToString() != expected) {
				mismatches++;
			}
		}
		INFO("pattern " << c.pattern << " options " << c.options);
		REQUIRE(mismatches == 0);
	}
}
