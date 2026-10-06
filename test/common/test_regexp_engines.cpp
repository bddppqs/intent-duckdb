#include "catch.hpp"
#include "duckdb.hpp"
#include "re2/re2.h"
#include "test_helpers.hpp"

#include <atomic>
#include <thread>

using namespace duckdb;
using duckdb_re2::RE2;
using duckdb_re2::StringPiece;

namespace {
//! xorshift64: a fixed sequence, so the generated strings are the same on every run
struct StringGenerator {
	uint64_t state;
	uint64_t Next() {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		return state;
	}
};

struct PatternCase {
	string pattern;
	string rewrite;
	string options;
};

//! The patterns: anchored and unanchored, captures and none, greedy and lazy repeats, alternations, character
//! classes with UTF-8 ranges, case folding, (?s) and (?m), and assertions RE2's tagged DFA does not take
const duckdb::vector<PatternCase> &Patterns() {
	static const duckdb::vector<PatternCase> cases {
	    {"^https?://([^/:?#]+)", "\\1", ""},
	    {"^https?://[^/]+(/[^?#]*)", "[\\1]", ""},
	    {"[?&]q=([^&]*)", "\\1", ""},
	    {"\\.([a-z]{2,6})(?:/|$)", "<\\1>", ""},
	    {"^([^=]+)=(.*)$", "\\2=\\1", ""},
	    {"^(.*)/(.*)$", "\\2|\\1", ""},
	    {"^(.).*$", "\\1", ""},
	    {"^(.).*$", "\\1", "s"},
	    {"(\\d+)$", "<\\1>", ""},
	    {"^(\\w+)", "\\1", ""},
	    {"(\\d{4})-(\\d{2})-(\\d{2})", "\\3.\\2.\\1", ""},
	    {"([a-z0-9._%+-]+)@([a-z0-9.-]+\\.[a-z]{2,})", "\\2", ""},
	    {"^(.*?)\\n.*$", "\\1", "s"},
	    {"(?s)^(a.*?b)(.*)$", "\\2\\1", ""},
	    {"<(.+?)>", "\\1", ""},
	    {"([а-яё]+)", "\\1", ""},
	    {"((a|b)+)(c?)", "\\1:\\2:\\3", ""},
	    {"(a|ab)(c|bcd)(d*)", "\\1,\\2,\\3", ""},
	    {"^(a+)(a*)$", "\\1|\\2", ""},
	    {"^(a*?)(a*)$", "\\1|\\2", ""},
	    {"^(?:(a)|b)*$", "\\1", ""},
	    {"(é+|я)(.?)", "\\2\\1", ""},
	    {"^(\\x{1F600}+)(.*)$", "\\2\\1", ""},
	    {"(A+)(b?)", "\\1-\\2", "i"},
	    {"^HTTPS?://WWW\\.([A-Z0-9.-]+)", "\\1", "i"},
	    {"^([^/]*)/.*\\z", "\\1", ""},
	    {"x*", "-", ""},
	    {"(x*)(y*)", "<\\1|\\2>", ""},
	    {"^$", "E", ""},
	    {"^(\\s*)(\\S+)(\\s*)$", "\\2", ""},
	    {"(\\pL+)", "\\1", ""},
	    {"([^a-z/.]{1,3})", "(\\1)", ""},
	    // assertions the tagged DFA does not take: RE2's other engines answer
	    {"(?m)^(\\w+)$", "<\\1>", ""},
	    {"\\b(ab|key)\\b", "[\\1]", ""},
	    {"^(\\w+)\\B(.*)$", "\\2", ""},
	};
	return cases;
}

duckdb::vector<string> Inputs() {
	// ASCII letters, digits and punctuation the patterns name, newlines and tabs, backslashes, UTF-8 sequences of two,
	// three and four bytes, and web address pieces
	const duckdb::vector<string> pieces {"a",  "b",  "c",   "x",   "y",       "z",  "A",  "B",   "0",       "7",
	                                     "/",  ":",  ".",   "=",   "@",       " ",  "-",  "_",   "\n",      "\t",
	                                     "\\", "é",  "я",   "ё",   "中",      "?",  "&",  "<",   ">",       "ab",
	                                     "\xF0\x9F\x98\x80", "www.", "key", "http://", "https://", "q=", "2024-05-17",
	                                     "bob@ex.org", "org/", "abcd"};
	StringGenerator generator {0x9E3779B97F4A7C15ULL};
	duckdb::vector<string> inputs {""};
	for (idx_t i = 0; i < 4000; i++) {
		string s;
		const idx_t length = generator.Next() % 24;
		for (idx_t k = 0; k < length; k++) {
			s += pieces[generator.Next() % pieces.size()];
		}
		inputs.push_back(s);
	}
	return inputs;
}

RE2::Options PatternOptions(const string &letters, bool tagged_dfa, int64_t max_mem = 0) {
	RE2::Options options;
	options.set_log_errors(false);
	for (auto letter : letters) {
		if (letter == 'i') {
			options.set_case_sensitive(false);
		} else if (letter == 's') {
			options.set_dot_nl(true);
		}
	}
	options.set_tagged_dfa(tagged_dfa);
	if (max_mem > 0) {
		options.set_max_mem(max_mem);
	}
	return options;
}

//! Every RE2 call the regular-expression functions make, and Match with every submatch at each anchoring, as one
//! string per input
string Results(const RE2 &re, const PatternCase &c, const string &input) {
	string result;
	string replaced = input;
	result += RE2::Replace(&replaced, re, c.rewrite) ? "1" : "0";
	result += replaced + "\x01";
	replaced = input;
	result += std::to_string(RE2::GlobalReplace(&replaced, re, c.rewrite)) + replaced + "\x01";
	string extracted;
	result += RE2::Extract(input, re, c.rewrite, &extracted) ? "1" : "0";
	result += extracted + "\x01";
	result += RE2::PartialMatch(input, re) ? "1" : "0";
	result += RE2::FullMatch(input, re) ? "1" : "0";
	const int nmatch = 1 + re.NumberOfCapturingGroups();
	duckdb::vector<StringPiece> match(nmatch);
	for (auto anchor : {RE2::UNANCHORED, RE2::ANCHOR_START, RE2::ANCHOR_BOTH}) {
		if (!re.Match(input, 0, input.size(), anchor, match.data(), nmatch)) {
			result += "-";
			continue;
		}
		for (auto &m : match) {
			result += m.data() ? std::to_string(m.data() - input.data()) + ":" + std::to_string(m.size()) : string("n");
			result += ",";
		}
	}
	return result;
}

idx_t CompareEngines(const duckdb::vector<string> &inputs, int64_t max_mem, idx_t &compiled) {
	idx_t mismatches = 0;
	for (auto &c : Patterns()) {
		RE2 reference(c.pattern, PatternOptions(c.options, false, max_mem));
		RE2 tagged(c.pattern, PatternOptions(c.options, true, max_mem));
		// a budget too small for the program fails both compilations alike
		REQUIRE(reference.ok() == tagged.ok());
		if (!reference.ok()) {
			continue;
		}
		compiled++;
		idx_t pattern_mismatches = 0;
		for (auto &input : inputs) {
			if (Results(reference, c, input) != Results(tagged, c, input)) {
				pattern_mismatches++;
			}
		}
		INFO("pattern " << c.pattern << " options " << c.options << " max_mem " << max_mem);
		CHECK(pattern_mismatches == 0);
		mismatches += pattern_mismatches;
	}
	return mismatches;
}
} // namespace

TEST_CASE("RE2 gives the same results with and without its tagged DFA", "[regexp]") {
	auto inputs = Inputs();
	REQUIRE(Patterns().size() >= 30);
	// the default memory budget, and budgets small enough that the tagged DFA runs out of states partway (or before
	// its first state) and RE2's other engines answer instead
	for (int64_t max_mem : {int64_t(0), int64_t(1) << 17, int64_t(1) << 16, int64_t(1) << 15}) {
		idx_t compiled = 0;
		REQUIRE(CompareEngines(inputs, max_mem, compiled) == 0);
		INFO("max_mem " << max_mem << " compiled " << compiled);
		REQUIRE(compiled >= (max_mem == 0 ? Patterns().size() : 20));
	}
}

TEST_CASE("Regular-expression functions agree with RE2 without its tagged DFA", "[regexp]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto inputs = Inputs();
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE inputs (id INTEGER, s VARCHAR)"));
	{
		Appender appender(con, "inputs");
		for (idx_t i = 0; i < inputs.size(); i++) {
			// VARCHAR values are valid UTF-8; the generated pieces are
			appender.AppendRow(Value::INTEGER(NumericCast<int32_t>(i)), Value(inputs[i]));
		}
	}
	for (auto &c : Patterns()) {
		RE2 reference(c.pattern, PatternOptions(c.options, false));
		REQUIRE(reference.ok());
		const string pattern = Value(c.pattern).ToSQLString();
		const string options = c.options.empty() ? string() : ", '" + c.options + "'";
		string sql = "SELECT regexp_replace(s, " + pattern + ", " + Value(c.rewrite).ToSQLString() + options +
		             "), regexp_matches(s, " + pattern + options + "), regexp_full_match(s, " + pattern + options + ")";
		if (reference.NumberOfCapturingGroups() >= 1) {
			sql += ", regexp_extract(s, " + pattern + ", 1" + options + ")";
		}
		sql += " FROM inputs ORDER BY id";
		auto result = con.Query(sql);
		REQUIRE_NO_FAIL(*result);
		REQUIRE(result->RowCount() == inputs.size());
		idx_t mismatches = 0;
		for (idx_t i = 0; i < inputs.size(); i++) {
			string replaced = inputs[i];
			RE2::Replace(&replaced, reference, c.rewrite);
			bool ok = result->GetValue(0, i).ToString() == replaced;
			ok = ok && result->GetValue(1, i).GetValue<bool>() == RE2::PartialMatch(inputs[i], reference);
			ok = ok && result->GetValue(2, i).GetValue<bool>() == RE2::FullMatch(inputs[i], reference);
			if (reference.NumberOfCapturingGroups() >= 1) {
				string extracted;
				RE2::Extract(inputs[i], reference, "\\1", &extracted);
				ok = ok && result->GetValue(3, i).ToString() == extracted;
			}
			mismatches += !ok;
		}
		INFO("pattern " << c.pattern << " options " << c.options);
		REQUIRE(mismatches == 0);
	}
}

TEST_CASE("RE2's tagged DFA is shared safely by concurrent searches", "[regexp]") {
	auto inputs = Inputs();
	const idx_t thread_count = 64;
	for (idx_t p = 0; p < 4; p++) {
		auto &c = Patterns()[p];
		RE2 reference(c.pattern, PatternOptions(c.options, false));
		// one compiled pattern, built lazily by whichever search holds it; the others run RE2's other engines
		RE2 shared(c.pattern, PatternOptions(c.options, true));
		REQUIRE(shared.ok());
		duckdb::vector<string> expected;
		for (auto &input : inputs) {
			expected.push_back(Results(reference, c, input));
		}
		std::atomic<idx_t> mismatches {0};
		duckdb::vector<std::thread> threads;
		for (idx_t t = 0; t < thread_count; t++) {
			threads.emplace_back([&, t]() {
				for (idx_t round = 0; round < 2; round++) {
					for (idx_t i = 0; i < inputs.size(); i++) {
						// each thread walks the inputs from its own offset
						const idx_t k = (i + t * 61) % inputs.size();
						if (Results(shared, c, inputs[k]) != expected[k]) {
							mismatches++;
						}
					}
				}
			});
		}
		for (auto &thread : threads) {
			thread.join();
		}
		INFO("pattern " << c.pattern);
		REQUIRE(mismatches.load() == 0);
	}
}
