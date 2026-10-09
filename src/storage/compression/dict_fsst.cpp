#include "duckdb/storage/compression/dict_fsst/common.hpp"
#include "duckdb/storage/compression/dict_fsst/analyze.hpp"
#include "duckdb/storage/compression/dict_fsst/compression.hpp"
#include "duckdb/storage/compression/dict_fsst/decompression.hpp"
#include "duckdb/storage/compression/dict_fsst/filter_verdict_cache.hpp"
#include "duckdb/storage/compression/dict_fsst/split_segment.hpp"
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"
#include "duckdb/storage/compression/dict_global/persisted_translation.hpp"
#include "duckdb/function/compression/compression.hpp"
#include "duckdb/function/compression_function.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__aarch64__) && !defined(__AARCH64EB__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#define DUCKDB_SINGLE_CODE_NEON 1
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

/*
Data layout per segment:
+-----------------------------------------------------+
|                  Header                             |
|   +---------------------------------------------+   |
|   |   dict_fsst_compression_header_t  header    |   |
|   +---------------------------------------------+   |
|                                                     |
+-----------------------------------------------------+
|             Selection Buffer               |
|   +------------------------------------+   |
|   |   uint16_t index_buffer_idx[]      |   |
|   +------------------------------------+   |
|      tuple index -> index buffer idx       |
|                                            |
+--------------------------------------------+
|               Index Buffer                 |
|   +------------------------------------+   |
|   |   uint16_t  dictionary_offset[]    |   |
|   +------------------------------------+   |
|  string_index -> offset in the dictionary  |
|                                            |
+--------------------------------------------+
|                Dictionary                  |
|   +------------------------------------+   |
|   |   uint8_t *raw_string_data         |   |
|   +------------------------------------+   |
|      the string data without lengths       |
|                                            |
+--------------------------------------------+
|             FSST Symbol Table (opt)        |
|   +------------------------------------+   |
|   |   duckdb_fsst_decoder_t table      |   |
|   +------------------------------------+   |
|                                            |
+--------------------------------------------+
*/

namespace duckdb {
namespace dict_fsst {

//===--------------------------------------------------------------------===//
// The per-segment dictionary skip (duckdb/common/tuning_defaults.hpp)
//===--------------------------------------------------------------------===//
//! The per-segment dictionary skip in the domain check
static bool DictionarySegmentSkipEnabled() {
	return kDictionarySegmentSkip;
}

//! The skip's eager completion of an incomplete or absent slot on the first ask (off: the lazy form)
static bool DictionarySegmentSkipEagerEnabled() {
	return kDictionarySegmentSkipEager;
}

//! The skip's reuse of its slot summary across scans (off: the fresh form, no slot summary read or written)
static bool DictionarySegmentSkipReuseEnabled() {
	return kDictionarySegmentSkipReuse;
}

//! The skip's ask of a negated canonical filter (off: a negated `contains` answers NO_PRUNING_POSSIBLE before the
//! segment's scan state, and so its dictionary block, is initialized)
static bool DictionarySegmentSkipNegatedEnabled() {
	return kDictionarySegmentSkipNegated;
}

struct DictFSSTCompressionStorage {
	static unique_ptr<AnalyzeState> StringInitAnalyze(ColumnData &col_data, PhysicalType type);
	static bool StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count);
	static idx_t StringFinalAnalyze(AnalyzeState &state_p);

	static unique_ptr<CompressionState> InitCompression(ColumnDataCheckpointData &checkpoint_data,
	                                                    unique_ptr<AnalyzeState> state);
	static void Compress(CompressionState &state_p, Vector &scan_vector, idx_t count);
	static void FinalizeCompress(CompressionState &state_p);

	static unique_ptr<SegmentScanState> StringInitScan(const QueryContext &context, ColumnSegment &segment);
	template <bool ALLOW_DICT_VECTORS>
	static void StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result,
	                              idx_t result_offset);
	static void StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count, Vector &result);
	static void StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id, Vector &result,
	                           idx_t result_idx);
};

//===--------------------------------------------------------------------===//
// Analyze
//===--------------------------------------------------------------------===//
unique_ptr<AnalyzeState> DictFSSTCompressionStorage::StringInitAnalyze(ColumnData &col_data, PhysicalType type) {
	auto &storage_manager = col_data.GetStorageManager();
	if (storage_manager.GetStorageVersion() < 5) {
		// dict_fsst not introduced yet, disable it
		return nullptr;
	}

	CompressionInfo info(col_data.GetBlockManager());
	return make_uniq<DictFSSTAnalyzeState>(info);
}

bool DictFSSTCompressionStorage::StringAnalyze(AnalyzeState &state_p, Vector &input, idx_t count) {
	auto &analyze_state = state_p.Cast<DictFSSTAnalyzeState>();
	return analyze_state.Analyze(input, count);
}

idx_t DictFSSTCompressionStorage::StringFinalAnalyze(AnalyzeState &state_p) {
	auto &analyze_state = state_p.Cast<DictFSSTAnalyzeState>();
	return analyze_state.FinalAnalyze();
}

//===--------------------------------------------------------------------===//
// Compress
//===--------------------------------------------------------------------===//
unique_ptr<CompressionState> DictFSSTCompressionStorage::InitCompression(ColumnDataCheckpointData &checkpoint_data,
                                                                         unique_ptr<AnalyzeState> state) {
	return make_uniq<DictFSSTCompressionState>(checkpoint_data,
	                                           unique_ptr_cast<AnalyzeState, DictFSSTAnalyzeState>(std::move(state)));
}

void DictFSSTCompressionStorage::Compress(CompressionState &state_p, Vector &scan_vector, idx_t count) {
	auto &state = state_p.Cast<DictFSSTCompressionState>();
	state.Compress(scan_vector, count);
}

void DictFSSTCompressionStorage::FinalizeCompress(CompressionState &state_p) {
	auto &state = state_p.Cast<DictFSSTCompressionState>();
	// the row group's last, partial vector is appended first (nothing is buffered when the writer is not aligned)
	state.FinalizeCompress();
}

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
unique_ptr<SegmentScanState> DictFSSTCompressionStorage::StringInitScan(const QueryContext &context,
                                                                        ColumnSegment &segment) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	// A codes-only scan reads the stored translation and the codes (a split segment's code block, else its own block)
	auto translation = dict_global::CodesOnlyTranslation(segment);
	if (translation) {
		if (dict_fsst::SegmentSplit(segment)) {
			auto state = make_uniq<CompressedStringScanState>(segment, BufferHandle());
			state->InitializeCodesOnly(std::move(translation));
			return std::move(state);
		}
		auto state = make_uniq<CompressedStringScanState>(segment, buffer_manager.Pin(segment.block));
		state->Initialize(false);
		if (state->mode == DictFSSTMode::FSST_ONLY || state->dict_count != translation->count) {
			throw IOException("A segment does not match its stored translation - the database file appears corrupted");
		}
		state->codes_only = true;
		state->global_translation = std::move(translation);
		return std::move(state);
	}
	auto state = make_uniq<CompressedStringScanState>(segment, buffer_manager.Pin(segment.block));
	state->Initialize(true);

	const auto &stats = segment.stats.statistics;
	if (stats.GetStatsType() == StatisticsType::STRING_STATS && StringStats::HasMaxStringLength(stats)) {
		state->all_values_inlined = StringStats::MaxStringLength(stats) <= string_t::INLINE_LENGTH;
	}
	return std::move(state);
}

//===--------------------------------------------------------------------===//
// Scan base data
//===--------------------------------------------------------------------===//
template <bool ALLOW_DICT_VECTORS>
void DictFSSTCompressionStorage::StringScanPartial(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count,
                                                   Vector &result, idx_t result_offset) {
	// clear any previously locked buffers and get the primary buffer handle
	auto &scan_state = state.scan_state->Cast<CompressedStringScanState>();

	auto start = state.GetPositionInSegment();
	if (scan_state.codes_only && (!ALLOW_DICT_VECTORS || !scan_state.AllowGlobalDictionaryScan(scan_count) ||
	                              result_offset != 0)) {
		// a vector that straddles a codes-only column's segments is emitted by TryScanGlobalDictionary
		throw InternalException("A codes-only scan of a DICT_FSST segment asked for strings");
	}
	if (ALLOW_DICT_VECTORS && scan_state.AllowGlobalDictionaryScan(scan_count)) {
		scan_state.ScanToDictionaryVector(segment, result, result_offset, start, scan_count);
	} else if (!ALLOW_DICT_VECTORS || !scan_state.AllowDictionaryScan(scan_count)) {
		scan_state.ScanToFlatVector(result, result_offset, start, scan_count, &state);
	} else {
		scan_state.ScanToDictionaryVector(segment, result, result_offset, start, scan_count);
	}
}

void DictFSSTCompressionStorage::StringScan(ColumnSegment &segment, ColumnScanState &state, idx_t scan_count,
                                            Vector &result) {
	StringScanPartial<true>(segment, state, scan_count, result, 0);
}

//===--------------------------------------------------------------------===//
// Fetch
//===--------------------------------------------------------------------===//
void DictFSSTCompressionStorage::StringFetchRow(ColumnSegment &segment, ColumnFetchState &state, row_t row_id,
                                                Vector &result, idx_t result_idx) {
	// fetch a single row from the string segment
	CompressedStringScanState scan_state(segment, state.GetOrInsertHandle(segment));
	scan_state.Initialize(false);
	scan_state.ScanToFlatVector(result, result_idx, NumericCast<idx_t>(row_id), 1);
}

//===--------------------------------------------------------------------===//
// Select
//===--------------------------------------------------------------------===//
void DictFSSTSelect(ColumnSegment &segment, ColumnScanState &state, idx_t vector_count, Vector &result,
                    const SelectionVector &sel, idx_t sel_count) {
	auto &scan_state = state.scan_state->Cast<CompressedStringScanState>();
	if (scan_state.codes_only) {
		auto &local = scan_state.GetSelVec(state.GetPositionInSegment(), vector_count);
		SelectionVector picked(MaxValue<idx_t>(sel_count, 1));
		for (idx_t i = 0; i < sel_count; i++) {
			picked.set_index(i, local.get_index(sel.get_index(i)));
		}
		scan_state.ScanToGlobalDictionary(picked, sel_count, result);
		return;
	}
	if (scan_state.mode == DictFSSTMode::FSST_ONLY) {
		// for FSST only
		auto start = state.GetPositionInSegment();
		scan_state.Select(result, start, sel, sel_count);
		return;
	}
	// fallback: scan + slice (the whole dictionary materialised first, as the plain scan does)
	scan_state.EnsureDictionary();
	DictFSSTCompressionStorage::StringScan(segment, state, vector_count, result);
	result.Slice(sel, sel_count);
}

//===--------------------------------------------------------------------===//
// Filter
//===--------------------------------------------------------------------===//
//! Apply a filter to the dictionary entries 1..dict_count-1 (slot zero represents NULL and is not necessarily
//! referenced by any row); dict_sel receives the entry indices relative to entry 1 that pass, the count is returned
static idx_t FilterDictionaryEntries(CompressedStringScanState &scan_state, const TableFilter &filter,
                                     TableFilterState &filter_state, SelectionVector &dict_sel) {
	idx_t non_null_count = scan_state.dict_count - 1;
	Vector dict_data(scan_state.dictionary->data, /*offset=*/1, scan_state.dict_count);
	UnifiedVectorFormat vdata;
	dict_data.ToUnifiedFormat(non_null_count, vdata);
	idx_t filter_count = non_null_count;
	ColumnSegment::FilterSelection(dict_sel, dict_data, vdata, filter, filter_state, non_null_count, filter_count);
	return filter_count;
}

//===--------------------------------------------------------------------===//
// Domain check: the pushed filter, including the TopN dynamic bound, evaluated on the dictionary entries
//===--------------------------------------------------------------------===//
//! AND a VARCHAR constant comparison over the dictionary entries into `pass`; slot zero (NULL) never passes
static bool ApplyConstantToDomain(CompressedStringScanState &scan_state, const ConstantFilter &filter, bool *pass) {
	auto matched = make_unsafe_uniq_array<bool>(scan_state.dict_count);
	std::fill(matched.get(), matched.get() + scan_state.dict_count, false);
	// an `=` / `<>` constant's verdicts from its slot on this dictionary when complete; else
	// evaluated below and published to the slot (nullptr when disabled, for another comparison or when not admitted)
	shared_ptr<FilterVerdictSlot> slot;
	if (ConstantFilterCacheEnabled()) {
		slot = AcquireFilterVerdictSlot(scan_state.segment, scan_state.dict_count, filter);
	}
	if (slot && slot->IsComplete()) {
		slot->ExpandNonNull(matched.get());
	} else {
		TableFilterState filter_state;
		SelectionVector dict_sel;
		idx_t count = FilterDictionaryEntries(scan_state, filter, filter_state, dict_sel);
		for (idx_t i = 0; i < count; i++) {
			matched[dict_sel.get_index(i) + 1] = true;
		}
		if (slot) {
			slot->SetAllNonNull(matched.get());
		}
	}
	for (idx_t i = 0; i < scan_state.dict_count; i++) {
		pass[i] = pass[i] && matched[i];
	}
	return true;
}

static void AppendDomainKey(string &key, const string *bound) {
	if (!bound) {
		key.push_back('\x01');
		return;
	}
	uint32_t length = static_cast<uint32_t>(bound->size());
	key.push_back('\x02');
	key.append(reinterpret_cast<const char *>(&length), sizeof(length));
	key.append(*bound);
}

//! Walk the supported filter shapes - constant comparisons, conjunctions (AND) of supported shapes and an optional
//! dynamic filter whose bound is read under its lock (an unset bound passes everything). Every dynamic bound is
//! appended to `key`; when `pass` is given the comparisons are evaluated over the dictionary. Returns false for an
//! unsupported shape, in which case nothing is decided.
static bool WalkDomainFilter(CompressedStringScanState &scan_state, const TableFilter &filter, string &key, bool *pass) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant_filter = filter.Cast<ConstantFilter>();
		if (constant_filter.constant.IsNull() ||
		    constant_filter.constant.type().InternalType() != PhysicalType::VARCHAR) {
			return false;
		}
		return !pass || ApplyConstantToDomain(scan_state, constant_filter, pass);
	}
	case TableFilterType::CONJUNCTION_AND: {
		for (auto &child : filter.Cast<ConjunctionAndFilter>().child_filters) {
			if (!WalkDomainFilter(scan_state, *child, key, pass)) {
				return false;
			}
		}
		return true;
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional_filter = filter.Cast<OptionalFilter>();
		if (!optional_filter.child_filter ||
		    optional_filter.child_filter->filter_type != TableFilterType::DYNAMIC_FILTER) {
			return false;
		}
		auto &dynamic_filter = optional_filter.child_filter->Cast<DynamicFilter>();
		auto &data = dynamic_filter.filter_data;
		if (!data) {
			AppendDomainKey(key, nullptr);
			return true;
		}
		const ConstantFilter *published;
		if (data->LoadPublished(published)) {
			// the immutable copy of the bound set last, read without the lock (kTopNBoundLockFree)
			if (!published) {
				AppendDomainKey(key, nullptr);
				return true;
			}
			auto &constant = published->constant;
			if (constant.IsNull() || constant.type().InternalType() != PhysicalType::VARCHAR) {
				return false;
			}
			AppendDomainKey(key, &StringValue::Get(constant));
			return !pass || ApplyConstantToDomain(scan_state, *published, pass);
		}
		ExpressionType comparison_type = ExpressionType::INVALID;
		Value bound;
		bool is_set = false;
		{
			lock_guard<mutex> l(data->lock);
			if (data->initialized && data->filter) {
				auto &constant = data->filter->constant;
				if (constant.IsNull() || constant.type().InternalType() != PhysicalType::VARCHAR) {
					return false;
				}
				AppendDomainKey(key, &StringValue::Get(constant));
				if (!pass) {
					return true;
				}
				comparison_type = data->filter->comparison_type;
				bound = constant;
				is_set = true;
			}
		}
		if (!is_set) {
			AppendDomainKey(key, nullptr);
			return true;
		}
		ConstantFilter bound_filter(comparison_type, std::move(bound));
		return ApplyConstantToDomain(scan_state, bound_filter, pass);
	}
	default:
		return false;
	}
}

//! FILTER_ALWAYS_FALSE when no dictionary entry of the segment satisfies the filter under the bounds of this moment;
//! the per-entry verdicts are cached on the scan state and reused while the bounds are unchanged (O(1) per vector)
static FilterPropagateResult CheckDictionaryDomain(CompressedStringScanState &scan_state, const TableFilter &filter) {
	if (scan_state.domain_filter == &filter) {
		if (scan_state.domain_unsupported) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
		if (scan_state.domain_valid) {
			scan_state.domain_probe.clear();
			if (WalkDomainFilter(scan_state, filter, scan_state.domain_probe, nullptr) &&
			    scan_state.domain_probe == scan_state.domain_key) {
				return scan_state.domain_qualifying == 0 ? FilterPropagateResult::FILTER_ALWAYS_FALSE
				                                         : FilterPropagateResult::NO_PRUNING_POSSIBLE;
			}
		}
	} else {
		scan_state.domain_filter = &filter;
		scan_state.domain_valid = false;
		scan_state.domain_unsupported = false;
	}
	// (re)compute the verdicts under the bounds of this moment; a bound only ever tightens, so verdicts computed
	// under an older bound stay a superset of the rows the consumer can still use
	if (!scan_state.domain_result) {
		scan_state.domain_result = make_unsafe_uniq_array<bool>(scan_state.dict_count);
	}
	auto pass = scan_state.domain_result.get();
	std::fill(pass, pass + scan_state.dict_count, true);
	scan_state.domain_valid = false;
	scan_state.domain_key.clear();
	if (!WalkDomainFilter(scan_state, filter, scan_state.domain_key, pass)) {
		scan_state.domain_unsupported = true;
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	idx_t qualifying = 0;
	for (idx_t i = 0; i < scan_state.dict_count; i++) {
		qualifying += pass[i] ? 1 : 0;
	}
	scan_state.domain_qualifying = qualifying;
	scan_state.domain_valid = true;
	return qualifying == 0 ? FilterPropagateResult::FILTER_ALWAYS_FALSE : FilterPropagateResult::NO_PRUNING_POSSIBLE;
}

static FilterPropagateResult DictFSSTCheckDomainBase(ColumnSegment &segment, ColumnScanState &state,
                                                     const TableFilter &filter) {
	if (segment.segment_type != ColumnSegmentType::PERSISTENT || !state.current) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (!state.initialized) {
		// the scan has not reached this segment yet - create its scan state exactly as the scan would
		segment.InitializeScan(state);
		state.internal_index = state.current->GetRowStart();
		state.initialized = true;
	}
	if (!state.scan_state) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto &scan_state = state.scan_state->Cast<CompressedStringScanState>();
	if (scan_state.codes_only) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (&scan_state.segment != &segment || scan_state.mode == DictFSSTMode::FSST_ONLY || scan_state.dict_count == 0) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (!scan_state.dictionary) {
		// materialise on demand only for a filter shape the domain walk supports (probed with a null pass
		// array, exactly as CheckDictionaryDomain probes an unchanged bound); an unsupported shape - the pushed
		// expression filters - answers NO_PRUNING_POSSIBLE without decoding the dictionary
		string probe;
		if (!WalkDomainFilter(scan_state, filter, probe, nullptr)) {
			scan_state.domain_filter = &filter;
			scan_state.domain_valid = false;
			scan_state.domain_unsupported = true;
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
		scan_state.EnsureDictionary();
		if (!scan_state.dictionary) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
	}
	return CheckDictionaryDomain(scan_state, filter);
}

//===--------------------------------------------------------------------===//
// The per-segment dictionary skip: a canonical `contains` / NOT `contains` filter whose segment dictionary holds
// no passing non-NULL entry answers FILTER_ALWAYS_FALSE, and the existing CheckZonemapSegments skips the vectors
//===--------------------------------------------------------------------===//
//! The canonical key of `filter` (empty when it is not canonical), resolved once per scan state and filter
static const string &SegmentSkipCanonicalKey(CompressedStringScanState &scan_state, const TableFilter &filter) {
	if (scan_state.segment_skip_filter != &filter) {
		scan_state.segment_skip_filter = &filter;
		if (!CanonicalFilterKey(filter, scan_state.segment_skip_key)) {
			scan_state.segment_skip_key.clear();
		}
	}
	return scan_state.segment_skip_key;
}

//! The passing non-NULL entries of the whole dictionary under `filter`, through ColumnSegment::FilterSelection as the
//! dense branch evaluates them; a deferred dictionary is decoded entry by entry into a scratch vector that is
//! discarded - no decoded dictionary is published and the scan state's own decode state is untouched. `pass`
//! (dict_count entries, or nullptr) receives each code's verdict; slot zero (NULL) never passes.
static idx_t SegmentSkipEvaluateDictionary(CompressedStringScanState &scan_state, const TableFilter &filter,
                                           TableFilterState &filter_state, bool *pass) {
	const idx_t dict_count = scan_state.dict_count;
	if (pass) {
		std::fill(pass, pass + dict_count, false);
	}
	if (dict_count < 2) {
		return 0;
	}
	const idx_t non_null_count = dict_count - 1;
	SelectionVector dict_sel;
	idx_t filter_count;
	if (scan_state.dictionary) {
		filter_count = FilterDictionaryEntries(scan_state, filter, filter_state, dict_sel);
	} else {
		Vector entries(LogicalType::VARCHAR, non_null_count);
		scan_state.DecodeEntriesInto(entries);
		UnifiedVectorFormat vdata;
		entries.ToUnifiedFormat(non_null_count, vdata);
		filter_count = non_null_count;
		ColumnSegment::FilterSelection(dict_sel, entries, vdata, filter, filter_state, non_null_count, filter_count);
	}
	if (pass) {
		for (idx_t i = 0; i < filter_count; i++) {
			pass[dict_sel.get_index(i) + 1] = true;
		}
	}
	return filter_count;
}

//! The segment skip for a canonical filter: FILTER_ALWAYS_FALSE when the segment's dictionary holds no passing non-NULL
//! entry - read from a complete slot's pass count, or (an incomplete or absent slot, with the eager completion enabled)
//! from one whole-dictionary evaluation that completes the slot; in the fresh form (no slot reuse) evaluated afresh on
//! every ask, the slot's summary neither read nor written and no verdict published. NO_PRUNING_POSSIBLE otherwise.
static FilterPropagateResult DictionarySegmentSkipCheckDomain(ColumnSegment &segment, ColumnScanState &state,
                                                          CompressedStringScanState &scan_state, const TableFilter &filter) {
	// the existing guards: a persistent DICT_FSST or DICTIONARY-mode segment with a dictionary
	if (&scan_state.segment != &segment || scan_state.mode == DictFSSTMode::FSST_ONLY || scan_state.dict_count == 0 ||
	    (!scan_state.dictionary && !scan_state.dictionary_deferred)) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto slot = AcquireFilterVerdictSlot(segment, scan_state.dict_count, filter);
	if (!slot) {
		// no slot admitted (the cache off, the cap, a fifth filter, no cache key): the standard path
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	const bool reuse = DictionarySegmentSkipReuseEnabled();
	idx_t pass_count;
	if (reuse && slot->IsComplete()) {
		pass_count = slot->PassCount();
	} else {
		if (reuse && !DictionarySegmentSkipEagerEnabled()) {
			// the lazy form: only a slot complete before the ask answers
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
		if (!state.context.Valid()) {
			return FilterPropagateResult::NO_PRUNING_POSSIBLE;
		}
		if (!scan_state.segment_skip_eval_state || scan_state.segment_skip_eval_filter != &filter) {
			scan_state.segment_skip_eval_state = TableFilterState::Initialize(*state.context.GetClientContext(), filter);
			scan_state.segment_skip_eval_filter = &filter;
		}
		if (reuse) {
			auto pass = make_unsafe_uniq_array<bool>(scan_state.dict_count);
			pass_count = SegmentSkipEvaluateDictionary(scan_state, filter, *scan_state.segment_skip_eval_state, pass.get());
			// every non-NULL code's verdict published to the slot, the pass count with it, the slot marked complete
			slot->SetAllNonNull(pass.get());
		} else {
			pass_count = SegmentSkipEvaluateDictionary(scan_state, filter, *scan_state.segment_skip_eval_state, nullptr);
		}
	}
	if (pass_count > 0) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	return FilterPropagateResult::FILTER_ALWAYS_FALSE;
}

//! Whether `filter` is a canonical `contains` under OPERATOR_NOT (the key's leading NOT bit)
static bool IsNegatedCanonicalFilter(const TableFilter &filter) {
	if (filter.filter_type != TableFilterType::EXPRESSION_FILTER ||
	    filter.Cast<ExpressionFilter>().expr->GetExpressionType() != ExpressionType::OPERATOR_NOT) {
		return false;
	}
	string key;
	return CanonicalFilterKey(filter, key) && !key.empty() && key[0] != '\x00';
}

static FilterPropagateResult DictFSSTCheckDomain(ColumnSegment &segment, ColumnScanState &state,
                                                 const TableFilter &filter) {
	const bool s1 = DictionarySegmentSkipEnabled();
	if (!s1) {
		return DictFSSTCheckDomainBase(segment, state, filter);
	}
	if (segment.segment_type != ColumnSegmentType::PERSISTENT || !state.current) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (!DictionarySegmentSkipNegatedEnabled() && IsNegatedCanonicalFilter(filter)) {
		// a negated filter prunes only a segment whose every entry holds the needle, and the domain walk does not
		// take an expression filter: answer before the scan state pins the segment's dictionary block
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (!state.initialized) {
		segment.InitializeScan(state);
		state.internal_index = state.current->GetRowStart();
		state.initialized = true;
	}
	if (!state.scan_state) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	auto &scan_state = state.scan_state->Cast<CompressedStringScanState>();
	if (scan_state.codes_only) {
		return FilterPropagateResult::NO_PRUNING_POSSIBLE;
	}
	if (!SegmentSkipCanonicalKey(scan_state, filter).empty()) {
		if (DictionarySegmentSkipCheckDomain(segment, state, scan_state, filter) == FilterPropagateResult::FILTER_ALWAYS_FALSE) {
			return FilterPropagateResult::FILTER_ALWAYS_FALSE;
		}
	}
	return DictFSSTCheckDomainBase(segment, state, filter);
}

//===--------------------------------------------------------------------===//
// Cross-request verdict cache: per-code verdicts of a canonical `contains` filter kept per dictionary
//===--------------------------------------------------------------------===//
//! The verdict slot of the pushed filter for this scan state's dictionary, resolved once per scan state and filter;
//! nullptr when the cache is off, the filter is not canonical or the slot is not admitted (the plain path)
static optional_ptr<FilterVerdictSlot> VerdictSlot(CompressedStringScanState &scan_state, const TableFilter &filter) {
	if (!FilterVerdictCacheEnabled()) {
		return nullptr;
	}
	if (scan_state.verdict_filter != &filter) {
		scan_state.verdict_filter = &filter;
		scan_state.verdict_slot = AcquireFilterVerdictSlot(scan_state.segment, scan_state.dict_count, filter);
	}
	return scan_state.verdict_slot.get();
}

//===--------------------------------------------------------------------===//
// Selection-aware filter: verdicts for the codes the selected rows reference, a flat survivor vector
//===--------------------------------------------------------------------===//
//! The selection-aware branch engages for an incoming selection of at most this many rows of a whole vector (half a
//! vector, the ratio DuckDB's executor-side dictionary optimisation uses as CHUNK_FILL_RATIO_THRESHOLD)
static constexpr idx_t SELECTION_AWARE_MAX_SEL_COUNT = 1024;

//! Evaluate the filter over the entries `codes` (a flat batch of decoded strings) as FilterDictionaryEntries does over
//! the whole dictionary, writing 1 (pass) or 2 (fail) into the verdict table; slot zero (NULL) is evaluated on its own
static void EvaluateCodes(CompressedStringScanState &scan_state, const vector<idx_t> &codes, const TableFilter &filter,
                          TableFilterState &filter_state) {
	auto verdicts = scan_state.verdicts.get();
	idx_t batch_count = 0;
	for (auto code : codes) {
		batch_count += code != 0 ? 1 : 0;
	}
	if (batch_count > 0) {
		Vector batch(LogicalType::VARCHAR, batch_count);
		auto batch_data = FlatVector::GetData<string_t>(batch);
		idx_t k = 0;
		string_t *dictionary_values = nullptr;
		if (scan_state.dictionary) {
			dictionary_values = FlatVector::GetData<string_t>(scan_state.dictionary->data);
		}
		for (auto code : codes) {
			if (code == 0) {
				continue;
			}
			batch_data[k++] = dictionary_values ? dictionary_values[code] : scan_state.DecodeEntry(code);
		}
		if (scan_state.dictionary) {
			StringVector::AddHeapReference(batch, scan_state.dictionary->data);
		} else if (scan_state.decode_buffer) {
			StringVector::AddHeapReference(batch, *scan_state.decode_buffer);
		}
		UnifiedVectorFormat vdata;
		batch.ToUnifiedFormat(batch_count, vdata);
		SelectionVector batch_sel;
		idx_t approved = batch_count;
		ColumnSegment::FilterSelection(batch_sel, batch, vdata, filter, filter_state, batch_count, approved);
		// the batch positions map to codes in gathering order
		vector<idx_t> positions;
		positions.reserve(batch_count);
		for (auto code : codes) {
			if (code != 0) {
				verdicts[code] = 2;
				positions.push_back(code);
			}
		}
		for (idx_t i = 0; i < approved; i++) {
			verdicts[positions[batch_sel.get_index(i)]] = 1;
		}
	}
	for (auto code : codes) {
		if (code != 0) {
			continue;
		}
		// Slot zero represents NULL: evaluate one NULL entry exactly as the whole-dictionary path does
		Vector null_data(LogicalType::VARCHAR, 1U);
		FlatVector::GetData<string_t>(null_data)[0] = string_t(nullptr, 0);
		FlatVector::Validity(null_data).SetInvalid(0);
		UnifiedVectorFormat null_vdata;
		null_data.ToUnifiedFormat(1, null_vdata);
		SelectionVector null_sel;
		idx_t null_filter_count = 1;
		ColumnSegment::FilterSelection(null_sel, null_data, null_vdata, filter, filter_state, 1, null_filter_count);
		verdicts[0] = null_filter_count == 1 ? 1 : 2;
	}
}

//! A sparse visit publishes the segment's decoded dictionary; off, the sparse path decodes on demand only
static bool PublishOnSparseVisitEnabled() {
	return kPublishDictionaryOnSparseVisit;
}

//! A sparse or straddling visit completes the segment's verdict slot; off, the slot is filled code by code
static bool VerdictSlotCompletionEnabled() {
	return kVerdictSlotCompletion;
}

//! Complete a segment's verdict slot at its first sparse or straddling visit with the dense path's whole-dictionary
//! evaluation (FilterDictionaryEntries, then SetAllNonNull), once per dictionary and filter, so later requests read every
//! code's verdict from the slot; only when the dictionary is materialised or its publication is admitted
static void CompleteVerdictSlot(CompressedStringScanState &scan_state, FilterVerdictSlot &slot, const TableFilter &filter,
                                TableFilterState &filter_state) {
	if (slot.IsComplete() || scan_state.mode != DictFSSTMode::DICT_FSST) {
		return;
	}
	if (!scan_state.dictionary) {
		if (!scan_state.dictionary_deferred || !scan_state.cache_dictionary_admitted || !PublishOnSparseVisitEnabled()) {
			return;
		}
		scan_state.EnsureDictionary();
		if (!scan_state.dictionary) {
			return;
		}
	}
	auto pass = make_unsafe_uniq_array<bool>(scan_state.dict_count);
	memset(pass.get(), 0, scan_state.dict_count * sizeof(bool));
	if (scan_state.dict_count > 1) {
		SelectionVector dict_sel;
		const idx_t filter_count = FilterDictionaryEntries(scan_state, filter, filter_state, dict_sel);
		for (idx_t i = 0; i < filter_count; i++) {
			pass[dict_sel.get_index(i) + 1] = true;
		}
	}
	slot.SetAllNonNull(pass.get());
}

static void SelectionAwareFilter(CompressedStringScanState &scan_state, idx_t start, idx_t vector_count, Vector &result,
                                 SelectionVector &sel, idx_t &sel_count, const TableFilter &filter,
                                 TableFilterState &filter_state) {
	D_ASSERT(result.GetVectorType() == VectorType::FLAT_VECTOR);
	// a segment the dense path would publish (DICT_FSST, persistent, admitted at Initialize) publishes its
	// whole dictionary on its first sparse visit too, so a later scan finds it cached; the rest of this function then
	// reads the decoded strings from the dictionary
	if (!scan_state.dictionary && scan_state.dictionary_deferred && scan_state.cache_dictionary_admitted &&
	    PublishOnSparseVisitEnabled()) {
		scan_state.EnsureDictionary();
	}
	const idx_t dict_count = scan_state.dict_count;
	if (!scan_state.verdicts) {
		scan_state.verdicts = make_unsafe_uniq_array<uint8_t>(dict_count);
		memset(scan_state.verdicts.get(), 0, dict_count);
	}
	auto verdicts = scan_state.verdicts.get();
	auto &dict_sel = scan_state.GetSelVec(start, vector_count);
	auto slot = VerdictSlot(scan_state, filter);
	if (slot && VerdictSlotCompletionEnabled()) {
		CompleteVerdictSlot(scan_state, *slot, filter, filter_state);
	}
	// gather the unknown codes among the selected rows, each once
	auto &pending = scan_state.pending_codes;
	pending.clear();
	for (idx_t idx = 0; idx < sel_count; idx++) {
		auto code = dict_sel.get_index(sel.get_index(idx));
		if (verdicts[code] == 0) {
			const auto known = slot ? slot->Get(code) : FilterVerdictSlot::UNKNOWN;
			if (known != FilterVerdictSlot::UNKNOWN) {
				verdicts[code] = known;
				continue;
			}
			verdicts[code] = 3;
			pending.push_back(code);
		}
	}
	if (!pending.empty()) {
		EvaluateCodes(scan_state, pending, filter, filter_state);
		if (slot) {
			for (auto code : pending) {
				slot->Set(code, verdicts[code] == 1);
			}
		}
	}
	// the flat survivor vector: every slot initialised (non-survivors are NULL empty strings), survivors carry their
	// decoded string; the selection rewritten to the passing rows
	auto result_data = FlatVector::GetData<string_t>(result);
	auto &validity = FlatVector::Validity(result);
	// an all-zero string_t is the empty inlined string (length 0, zeroed prefix and inline bytes)
	memset(result_data, 0, vector_count * sizeof(string_t));
	validity.SetAllInvalid(vector_count);
	string_t *dictionary_values = nullptr;
	if (scan_state.dictionary) {
		dictionary_values = FlatVector::GetData<string_t>(scan_state.dictionary->data);
	}
	SelectionVector new_sel(sel_count);
	idx_t approved_tuple_count = 0;
	for (idx_t idx = 0; idx < sel_count; idx++) {
		auto row_idx = sel.get_index(idx);
		auto code = dict_sel.get_index(row_idx);
		if (verdicts[code] != 1) {
			continue;
		}
		new_sel.set_index(approved_tuple_count++, row_idx);
		if (code != 0) {
			result_data[row_idx] = dictionary_values ? dictionary_values[code]
			                       : slot            ? scan_state.DecodeEntry(code)
			                                         : scan_state.decoded[code];
			validity.SetValid(row_idx);
		}
	}
	if (approved_tuple_count < vector_count) {
		sel.Initialize(new_sel);
	}
	sel_count = approved_tuple_count;
	if (scan_state.dictionary) {
		StringVector::AddHeapReference(result, scan_state.dictionary->data);
	} else {
		scan_state.decoded_on_demand = true;
		if (scan_state.decode_buffer) {
			StringVector::AddHeapReference(result, *scan_state.decode_buffer);
		}
	}
}

//===--------------------------------------------------------------------===//
// The verdict-selected straddle filter
//===--------------------------------------------------------------------===//
// A vector that crosses a segment boundary is scanned flat by ColumnData::Filter and its filter evaluated row by row. The
// flat value of a row of a DICT_FSST piece is its code's dictionary entry (ScanToFlatVector over GetSelVec's codes), and a
// slot verdict is the filter's value on that entry, so such a row takes its code's verdict. A row with no verdict -
// code 0, an invalid row, an unknown code, a piece without a slot or not DICT_FSST - is evaluated by FilterSelection as
// before; the survivors keep the selection's order.
static constexpr idx_t STRADDLE_MAX_PIECES = 16;
static constexpr uint8_t STRADDLE_EVAL_FAIL = 3;
static constexpr uint8_t STRADDLE_EVAL_PASS = 4;

bool StraddleVerdictFilter(SegmentScanState *const *piece_states, const idx_t *piece_starts, const idx_t *piece_counts,
                           idx_t piece_count, Vector &result, UnifiedVectorFormat &vdata, SelectionVector &sel,
                           idx_t scan_count, idx_t &sel_count, const TableFilter &filter, TableFilterState &filter_state) {
	if (!FilterVerdictCacheEnabled() || piece_count == 0 || piece_count > STRADDLE_MAX_PIECES ||
	    scan_count > STANDARD_VECTOR_SIZE || sel_count == 0) {
		return false;
	}
	idx_t total = 0;
	for (idx_t p = 0; p < piece_count; p++) {
		total += piece_counts[p];
	}
	if (total != scan_count) {
		return false;
	}
	// per row: the verdict (the slot's 0 unknown, 1 pass, 2 fail; then 3 / 4 evaluated here), the code to publish under
	// (0: nothing to publish) and its piece
	uint8_t row_verdict[STANDARD_VECTOR_SIZE];
	uint32_t row_code[STANDARD_VECTOR_SIZE];
	uint8_t row_piece[STANDARD_VECTOR_SIZE];
	FilterVerdictSlot *slots[STRADDLE_MAX_PIECES];
	memset(row_verdict, 0, scan_count);
	memset(row_code, 0, scan_count * sizeof(uint32_t));
	bool any_slot = false;
	idx_t offset = 0;
	for (idx_t p = 0; p < piece_count; p++) {
		slots[p] = nullptr;
		const idx_t count = piece_counts[p];
		auto piece_state = piece_states[p];
		if (piece_state && count > 0) {
			auto &scan_state = piece_state->Cast<CompressedStringScanState>();
			if (scan_state.mode == DictFSSTMode::DICT_FSST) {
				auto slot = VerdictSlot(scan_state, filter);
				if (slot) {
					if (VerdictSlotCompletionEnabled()) {
						CompleteVerdictSlot(scan_state, *slot, filter, filter_state);
					}
					slots[p] = slot.get();
					any_slot = true;
					auto &codes = scan_state.GetSelVec(piece_starts[p], count);
					for (idx_t i = 0; i < count; i++) {
						const auto code = codes.get_index(i);
						if (code != 0 && code < slot->dict_count) {
							row_code[offset + i] = UnsafeNumericCast<uint32_t>(code);
							row_verdict[offset + i] = slot->Get(code);
						}
					}
				}
			}
		}
		memset(row_piece + offset, UnsafeNumericCast<uint8_t>(p), count);
		offset += count;
	}
	if (!any_slot) {
		return false;
	}
	if (!vdata.validity.AllValid()) {
		for (idx_t row = 0; row < scan_count; row++) {
			if (!vdata.validity.RowIsValid(vdata.sel->get_index(row))) {
				row_verdict[row] = FilterVerdictSlot::UNKNOWN;
				row_code[row] = 0;
			}
		}
	}
	// the selected rows without a verdict, evaluated exactly as before (in selection order)
	sel_t unknown_data[STANDARD_VECTOR_SIZE];
	SelectionVector unknown_sel(unknown_data);
	idx_t unknown_count = 0;
	for (idx_t idx = 0; idx < sel_count; idx++) {
		const auto row = sel.get_index(idx);
		if (row_verdict[row] == FilterVerdictSlot::UNKNOWN) {
			row_verdict[row] = STRADDLE_EVAL_FAIL;
			unknown_sel.set_index(unknown_count++, row);
		}
	}
	if (unknown_count > 0) {
		idx_t approved = unknown_count;
		ColumnSegment::FilterSelection(unknown_sel, result, vdata, filter, filter_state, scan_count, approved);
		for (idx_t i = 0; i < approved; i++) {
			row_verdict[unknown_sel.get_index(i)] = STRADDLE_EVAL_PASS;
		}
	}
	SelectionVector new_sel(sel_count);
	idx_t approved_tuple_count = 0;
	for (idx_t idx = 0; idx < sel_count; idx++) {
		const auto row = sel.get_index(idx);
		const auto verdict = row_verdict[row];
		const bool pass = verdict == FilterVerdictSlot::PASS || verdict == STRADDLE_EVAL_PASS;
		if (verdict >= STRADDLE_EVAL_FAIL && row_code[row] != 0) {
			slots[row_piece[row]]->Set(row_code[row], pass);
		}
		if (pass) {
			new_sel.set_index(approved_tuple_count++, row);
		}
	}
	sel.Initialize(new_sel);
	sel_count = approved_tuple_count;
	return true;
}

//===--------------------------------------------------------------------===//
// The single-code filter fast path of the whole-dictionary branch
//===--------------------------------------------------------------------===//
// When the filter's verdicts over the dictionary entries 1..dict_count-1 leave exactly one failing entry (a <> whose
// constant is in the dictionary) or exactly one passing entry (an =), the survivors of a whole unselected vector are a
// comparison of each code with that one code, written without a branch or a load from the verdict array.
// Off, the verdict loop runs instead.
static bool SingleCodeFastPathEnabled() {
	return kSingleCodeFilterFastPath;
}

static constexpr uint8_t SINGLE_CODE_NONE = 0;
static constexpr uint8_t SINGLE_CODE_ONE_FAILS = 1;
static constexpr uint8_t SINGLE_CODE_ONE_PASSES = 2;

//===--------------------------------------------------------------------===//
// The single-code fast path's block loop
//===--------------------------------------------------------------------===//
// On a whole unselected vector that starts on a 32-row boundary, the dense branch's single-code path runs per 32-code
// block: the block skip compares the block's packed bytes with the packed image of 32 copies of the one failing code
// and, when they are equal, writes the block's 32 dict_sel slots with that code without unpacking it (no row of it
// survives); the block mask unpacks any other block and writes its survivors from a 32-bit mask. Each member has its
// compile-time default (both off: the per-code single-code path).
static bool SingleCodeFilterBlockMaskEnabled() {
#if defined(__SSE2__)
	return kSingleCodeFilterBlockMask;
#elif defined(DUCKDB_SINGLE_CODE_NEON)
	return kSingleCodeFilterBlockMask && kSingleCodeFilterBlockMaskNeon;
#else
	return false;
#endif
}

static bool SingleCodeFilterBlockSkipEnabled() {
	return kSingleCodeFilterBlockSkip;
}

#if defined(__SSE2__) || defined(DUCKDB_SINGLE_CODE_NEON)
// The block mask's survivor extraction: the block's 32-bit mask by SSE2 compares
// of its 32 unpacked codes (four uint32_t lanes per compare on the generic x86-64 target: no -march, no runtime
// dispatch, no __AVX2__ path), packed to 32 bits; the survivors written per 8-bit mask group from a 256-entry table of 8
// uint32_t indices (sel_t is uint32_t: 32 bytes per entry, never a byte table) with the group's base index added by
// _mm_add_epi32, two unconditional 128-bit stores per group, the output advanced by the group's popcount (read from a
// 256-entry count table beside the index table: __builtin_popcount is a libgcc call at this target); no per-survivor
// branch, the survivors in ascending order. The stores stay inside a STANDARD_VECTOR_SIZE output: before a group the
// output holds at most the rows before it, so its eight slots end at or before the vector's last slot. On aarch64 the
// Advanced SIMD twins below compute the same mask and write the same output.
struct alignas(16) SingleCodeBlockTable {
	uint32_t index[256][8]; // the set lanes of each 8-bit mask, ascending, zero-padded to 8
	uint8_t count[256];     // the popcount of each 8-bit mask
};

static SingleCodeBlockTable SingleCodeBuildBlockTable() {
	SingleCodeBlockTable table;
	for (uint32_t m = 0; m < 256; m++) {
		uint32_t n = 0;
		for (uint32_t b = 0; b < 8; b++) {
			if (m & (1u << b)) {
				table.index[m][n++] = b;
			}
		}
		table.count[m] = uint8_t(n);
		for (; n < 8; n++) {
			table.index[m][n] = 0;
		}
	}
	return table;
}

static const SingleCodeBlockTable &SingleCodeGetBlockTable() {
	static const SingleCodeBlockTable table = SingleCodeBuildBlockTable();
	return table;
}
#endif

#if defined(__SSE2__)
// ONE_FAILS: lane i set iff code i passes the per-code term, code != c && (code != 0 || null_passes); fails_zero is
// all-ones when null_passes is 0 (the NULL slot's code 0 fails) and zero otherwise
static inline uint32_t SingleCodeFailsMask(const uint32_t *codes, const __m128i vc, const __m128i fails_zero) {
	const __m128i zero = _mm_setzero_si128();
	__m128i fail[8];
	for (uint32_t q = 0; q < 8; q++) {
		const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codes + 4 * q));
		fail[q] = _mm_or_si128(_mm_cmpeq_epi32(v, vc), _mm_and_si128(_mm_cmpeq_epi32(v, zero), fails_zero));
	}
	const __m128i lo = _mm_packs_epi16(_mm_packs_epi32(fail[0], fail[1]), _mm_packs_epi32(fail[2], fail[3]));
	const __m128i hi = _mm_packs_epi16(_mm_packs_epi32(fail[4], fail[5]), _mm_packs_epi32(fail[6], fail[7]));
	return ~(uint32_t(_mm_movemask_epi8(lo)) | (uint32_t(_mm_movemask_epi8(hi)) << 16));
}

// ONE_PASSES: lane i set iff code i passes the per-code term, code == c || (code == 0 && null_passes); passes_zero is
// all-ones when null_passes is 1 and zero otherwise
static inline uint32_t SingleCodePassesMask(const uint32_t *codes, const __m128i vc, const __m128i passes_zero) {
	const __m128i zero = _mm_setzero_si128();
	__m128i pass[8];
	for (uint32_t q = 0; q < 8; q++) {
		const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(codes + 4 * q));
		pass[q] = _mm_or_si128(_mm_cmpeq_epi32(v, vc), _mm_and_si128(_mm_cmpeq_epi32(v, zero), passes_zero));
	}
	const __m128i lo = _mm_packs_epi16(_mm_packs_epi32(pass[0], pass[1]), _mm_packs_epi32(pass[2], pass[3]));
	const __m128i hi = _mm_packs_epi16(_mm_packs_epi32(pass[4], pass[5]), _mm_packs_epi32(pass[6], pass[7]));
	return uint32_t(_mm_movemask_epi8(lo)) | (uint32_t(_mm_movemask_epi8(hi)) << 16);
}

// the survivors of one block (base: its first row) written at out; returns the output's new end
static inline uint32_t *SingleCodeExtractBlock(const SingleCodeBlockTable &table, uint32_t mask, uint32_t base, uint32_t *out) {
	for (uint32_t g = 0; g < 4; g++) {
		const uint32_t m8 = (mask >> (8 * g)) & 0xFFu;
		const __m128i add = _mm_set1_epi32(int(base + 8 * g));
		const __m128i *entry = reinterpret_cast<const __m128i *>(table.index[m8]);
		_mm_storeu_si128(reinterpret_cast<__m128i *>(out), _mm_add_epi32(_mm_load_si128(entry), add));
		_mm_storeu_si128(reinterpret_cast<__m128i *>(out + 4), _mm_add_epi32(_mm_load_si128(entry + 1), add));
		out += table.count[m8];
	}
	return out;
}
#endif

#if defined(DUCKDB_SINGLE_CODE_NEON)
// The Advanced SIMD twin of the SSE2 helpers above: the same masks and the same output, bit for bit. The 32 compare
// results (all-ones or zero per uint32_t lane) narrow to 32 bytes, each byte keeps its lane's bit within its group of 8,
// and three pairwise adds sum each group of 8 bytes into one mask byte (the bits are distinct, so no carry).
static inline uint32_t SingleCodeLaneMask(const uint32x4_t lanes[8]) {
	const uint8x16_t lo = vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(lanes[0]), vmovn_u32(lanes[1]))),
	                                  vmovn_u16(vcombine_u16(vmovn_u32(lanes[2]), vmovn_u32(lanes[3]))));
	const uint8x16_t hi = vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(lanes[4]), vmovn_u32(lanes[5]))),
	                                  vmovn_u16(vcombine_u16(vmovn_u32(lanes[6]), vmovn_u32(lanes[7]))));
	static const uint8_t BIT_WEIGHTS[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
	const uint8x16_t weights = vld1q_u8(BIT_WEIGHTS);
	uint8x16_t sum = vpaddq_u8(vandq_u8(lo, weights), vandq_u8(hi, weights));
	sum = vpaddq_u8(sum, sum);
	sum = vpaddq_u8(sum, sum);
	return vgetq_lane_u32(vreinterpretq_u32_u8(sum), 0);
}

// ONE_FAILS (as the SSE2 form): lane i set iff code != c && (code != 0 || null_passes)
static inline uint32_t SingleCodeFailsMask(const uint32_t *codes, const uint32x4_t vc, const uint32x4_t fails_zero) {
	const uint32x4_t zero = vdupq_n_u32(0);
	uint32x4_t fail[8];
	for (uint32_t q = 0; q < 8; q++) {
		const uint32x4_t v = vld1q_u32(codes + 4 * q);
		fail[q] = vorrq_u32(vceqq_u32(v, vc), vandq_u32(vceqq_u32(v, zero), fails_zero));
	}
	return ~SingleCodeLaneMask(fail);
}

// ONE_PASSES (as the SSE2 form): lane i set iff code == c || (code == 0 && null_passes)
static inline uint32_t SingleCodePassesMask(const uint32_t *codes, const uint32x4_t vc, const uint32x4_t passes_zero) {
	const uint32x4_t zero = vdupq_n_u32(0);
	uint32x4_t pass[8];
	for (uint32_t q = 0; q < 8; q++) {
		const uint32x4_t v = vld1q_u32(codes + 4 * q);
		pass[q] = vorrq_u32(vceqq_u32(v, vc), vandq_u32(vceqq_u32(v, zero), passes_zero));
	}
	return SingleCodeLaneMask(pass);
}

// the survivors of one block (base: its first row) written at out; returns the output's new end
static inline uint32_t *SingleCodeExtractBlock(const SingleCodeBlockTable &table, uint32_t mask, uint32_t base, uint32_t *out) {
	for (uint32_t g = 0; g < 4; g++) {
		const uint32_t m8 = (mask >> (8 * g)) & 0xFFu;
		const uint32x4_t add = vdupq_n_u32(base + 8 * g);
		vst1q_u32(out, vaddq_u32(vld1q_u32(table.index[m8]), add));
		vst1q_u32(out + 4, vaddq_u32(vld1q_u32(table.index[m8] + 4), add));
		out += table.count[m8];
	}
	return out;
}
#endif

//! The packed image of 32 copies of the one failing code at the scan state's index width (PackBlock, the form
//! the segment's indices were packed in: PackBuffer<sel_t> per 32-code group)
static void SingleCodeBuildImage(CompressedStringScanState &scan_state) {
	sel_t copies[BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE];
	for (idx_t i = 0; i < BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE; i++) {
		copies[i] = UnsafeNumericCast<sel_t>(scan_state.single_code);
	}
	memset(scan_state.single_code_image, 0, sizeof(scan_state.single_code_image));
	BitpackingPrimitives::PackBlock<sel_t>(data_ptr_cast(scan_state.single_code_image), copies,
	                                       scan_state.dictionary_indices_width);
	scan_state.single_code_image_valid = true;
}

//! Classify filter_result over the entries 1..dict_count-1 (computed just before, `filter_count` of them passing and
//! listed by `dict_sel` relative to entry 1)
static void ClassifySingleCode(CompressedStringScanState &scan_state, const SelectionVector &dict_sel,
                               idx_t filter_count) {
	scan_state.single_code_shape = SINGLE_CODE_NONE;
	scan_state.single_code_image_valid = false;
	if (scan_state.dict_count < 2) {
		return;
	}
	const idx_t non_null_count = scan_state.dict_count - 1;
	if (non_null_count - filter_count == 1) {
		for (idx_t idx = 1; idx < scan_state.dict_count; idx++) {
			if (!scan_state.filter_result[idx]) {
				scan_state.single_code = UnsafeNumericCast<uint32_t>(idx);
				scan_state.single_code_shape = SINGLE_CODE_ONE_FAILS;
				if (scan_state.dictionary_indices_width > 0 && scan_state.dictionary_indices_width <= 32) {
					SingleCodeBuildImage(scan_state);
				}
				return;
			}
		}
	} else if (filter_count == 1) {
		scan_state.single_code = UnsafeNumericCast<uint32_t>(dict_sel.get_index(0) + 1);
		scan_state.single_code_shape = SINGLE_CODE_ONE_PASSES;
	}
}

//! ClassifySingleCode over filter_result filled from a complete constant-comparison slot
//! (`filter_count` of the entries 1..dict_count-1 passing), the one passing entry found in filter_result instead of
//! dict_sel
static void ClassifySingleCodeFromVerdicts(CompressedStringScanState &scan_state, idx_t filter_count) {
	scan_state.single_code_shape = SINGLE_CODE_NONE;
	scan_state.single_code_image_valid = false;
	if (scan_state.dict_count < 2) {
		return;
	}
	const idx_t non_null_count = scan_state.dict_count - 1;
	const bool one_fails = non_null_count - filter_count == 1;
	if (!one_fails && filter_count != 1) {
		return;
	}
	for (idx_t idx = 1; idx < scan_state.dict_count; idx++) {
		if (scan_state.filter_result[idx] != one_fails) {
			scan_state.single_code = UnsafeNumericCast<uint32_t>(idx);
			if (one_fails) {
				scan_state.single_code_shape = SINGLE_CODE_ONE_FAILS;
				if (scan_state.dictionary_indices_width > 0 && scan_state.dictionary_indices_width <= 32) {
					SingleCodeBuildImage(scan_state);
				}
			} else {
				scan_state.single_code_shape = SINGLE_CODE_ONE_PASSES;
			}
			return;
		}
	}
}

//! The NULL slot's verdict, evaluated once per scan state exactly as the verdict loop evaluates it on first use
static void EnsureNullVerdict(CompressedStringScanState &scan_state, const TableFilter &filter,
                              TableFilterState &filter_state) {
	if (scan_state.null_filter_result_initialized) {
		return;
	}
	Vector null_data(scan_state.dictionary->data, /*offset=*/0, /*end=*/1);
	UnifiedVectorFormat null_vdata;
	null_data.ToUnifiedFormat(1, null_vdata);
	SelectionVector null_sel;
	idx_t null_filter_count = 1;
	ColumnSegment::FilterSelection(null_sel, null_data, null_vdata, filter, filter_state, 1, null_filter_count);
	scan_state.filter_result[0] = null_filter_count == 1;
	scan_state.null_filter_result_initialized = true;
}

//! The survivors of `count` rows with codes `dict_sel` under the single-code shape: with one failing entry c a row
//! passes iff code != c and (code != 0 or the NULL slot passes); with one passing entry c iff code == c or (code == 0
//! and the NULL slot passes).  Every row index is written; the count advances only for a survivor.
static idx_t SingleCodeSurvivors(const CompressedStringScanState &scan_state, const SelectionVector &dict_sel,
                                 idx_t count, bool null_slot_passes, SelectionVector &new_sel) {
	const sel_t *codes = dict_sel.data();
	const auto c = UnsafeNumericCast<sel_t>(scan_state.single_code);
	const idx_t null_passes = null_slot_passes ? 1 : 0;
	sel_t *out = new_sel.data();
	idx_t approved = 0;
	if (scan_state.single_code_shape == SINGLE_CODE_ONE_FAILS) {
		for (idx_t i = 0; i < count; i++) {
			const sel_t code = codes[i];
			out[approved] = UnsafeNumericCast<sel_t>(i);
			approved += static_cast<idx_t>(code != c) & (static_cast<idx_t>(code != 0) | null_passes);
		}
	} else {
		for (idx_t i = 0; i < count; i++) {
			const sel_t code = codes[i];
			out[approved] = UnsafeNumericCast<sel_t>(i);
			approved += static_cast<idx_t>(code == c) | (static_cast<idx_t>(code == 0) & null_passes);
		}
	}
	return approved;
}

//! The single-code path also runs under a domain verdict of the same shape; off, the verdict loop runs under a domain
//! verdict
static bool SingleCodeFilterWithVerdictsEnabled() {
	return kSingleCodeFilterWithVerdicts;
}

//! Whether the domain verdicts over the entries 1..dict_count-1 have filter_result's single-code shape with the
//! same code (that one entry failing and every other passing, or that one passing and every other failing), so that
//! SingleCodeSurvivors with the domain's NULL-slot verdict selects exactly the rows the domain verdict loop selects
static bool DomainKeepsSingleCode(const CompressedStringScanState &scan_state) {
	const bool *domain = scan_state.domain_result.get();
	const idx_t non_null_passing = scan_state.domain_qualifying - (domain[0] ? 1 : 0);
	if (scan_state.single_code_shape == SINGLE_CODE_ONE_FAILS) {
		return non_null_passing + 2 == scan_state.dict_count && !domain[scan_state.single_code];
	}
	return non_null_passing == 1 && domain[scan_state.single_code];
}

//! Whether this call takes the block loop - the single-code guard DictFSSTFilter computes (with the same-shape domain
//! verdict), a whole vector starting on a 32-row boundary, and the block mask enabled, or the block skip enabled on the
//! ONE_FAILS shape
static bool SingleCodeBlockLoop(const CompressedStringScanState &scan_state, bool single_code, idx_t vector_count,
                                idx_t start) {
	if (!single_code) {
		return false;
	}
	if (vector_count != STANDARD_VECTOR_SIZE || start % BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE != 0) {
		return false;
	}
	return SingleCodeFilterBlockMaskEnabled() ||
	       (SingleCodeFilterBlockSkipEnabled() && scan_state.single_code_shape == SINGLE_CODE_ONE_FAILS && scan_state.single_code_image_valid);
}

//! The dict_sel buffer the block loop fills (GetSelVec's own buffer at a group-aligned start, never unpacked here)
static const SelectionVector &SingleCodeBlockSelVec(CompressedStringScanState &scan_state) {
	if (!scan_state.sel_vec || scan_state.sel_vec_size < STANDARD_VECTOR_SIZE) {
		scan_state.sel_vec_size = STANDARD_VECTOR_SIZE;
		scan_state.sel_vec = make_buffer<SelectionVector>(STANDARD_VECTOR_SIZE);
	}
	return *scan_state.sel_vec;
}

//! The survivors of a whole vector under the single-code shape, one 32-code block at a time. The block skip (ONE_FAILS
//! only) skips a block whose packed bytes equal the image of 32 copies of the failing code, writing its 32 dict_sel slots
//! with that code; every other block is unpacked into its dict_sel slots and its survivors written by the block mask
//! (or, with the mask off, by the per-code term). The survivors and the dict_sel slots of the unpacked blocks are the
//! per-code path's exactly.
static idx_t SingleCodeBlockSurvivors(CompressedStringScanState &scan_state, idx_t start, bool null_slot_passes,
                                      SelectionVector &new_sel) {
	static constexpr idx_t BLOCK = BitpackingPrimitives::BITPACKING_ALGORITHM_GROUP_SIZE;
	const auto width = scan_state.dictionary_indices_width;
	const idx_t block_bytes = (BLOCK * width) / 8;
	const data_ptr_t packed = &scan_state.dictionary_indices_ptr[(start * width) / 8];
	sel_t *const codes = scan_state.sel_vec->data();
	const auto c = UnsafeNumericCast<sel_t>(scan_state.single_code);
	const bool fails = scan_state.single_code_shape == SINGLE_CODE_ONE_FAILS;
	const bool skip = fails && SingleCodeFilterBlockSkipEnabled() && scan_state.single_code_image_valid;
	const bool null_passes = null_slot_passes;
	sel_t *const out = new_sel.data();
	sel_t *end = out;
#if defined(__SSE2__)
	const bool mask_scan = SingleCodeFilterBlockMaskEnabled();
	const SingleCodeBlockTable &table = SingleCodeGetBlockTable();
	const __m128i vc = _mm_set1_epi32(int(c));
	// ONE_FAILS: all-ones iff the NULL slot fails; ONE_PASSES: all-ones iff it passes
	const __m128i zero_term = _mm_set1_epi32(fails ? (null_passes ? 0 : -1) : (null_passes ? -1 : 0));
#elif defined(DUCKDB_SINGLE_CODE_NEON)
	const bool mask_scan = SingleCodeFilterBlockMaskEnabled();
	const SingleCodeBlockTable &table = SingleCodeGetBlockTable();
	const uint32x4_t vc = vdupq_n_u32(c);
	// ONE_FAILS: all-ones iff the NULL slot fails; ONE_PASSES: all-ones iff it passes
	const uint32x4_t zero_term = vdupq_n_u32(fails ? (null_passes ? 0u : ~0u) : (null_passes ? ~0u : 0u));
#else
	const bool mask_scan = false;
#endif
	for (idx_t block = 0; block < STANDARD_VECTOR_SIZE / BLOCK; block++) {
		const data_ptr_t block_packed = packed + block * block_bytes;
		sel_t *const slots = codes + block * BLOCK;
		if (skip && memcmp(block_packed, scan_state.single_code_image, block_bytes) == 0) {
			for (idx_t i = 0; i < BLOCK; i++) {
				slots[i] = c;
			}
			continue;
		}
		BitpackingPrimitives::UnPackBlock<sel_t>(data_ptr_cast(slots), block_packed, width);
		const auto base = UnsafeNumericCast<sel_t>(block * BLOCK);
#if defined(__SSE2__) || defined(DUCKDB_SINGLE_CODE_NEON)
		if (mask_scan) {
			const uint32_t mask = fails ? SingleCodeFailsMask(slots, vc, zero_term) : SingleCodePassesMask(slots, vc, zero_term);
			end = SingleCodeExtractBlock(table, mask, base, end);
			continue;
		}
#endif
		const idx_t null_term = null_passes ? 1 : 0;
		for (idx_t i = 0; i < BLOCK; i++) {
			const sel_t code = slots[i];
			*end = UnsafeNumericCast<sel_t>(base + i);
			if (fails) {
				end += static_cast<idx_t>(code != c) & (static_cast<idx_t>(code != 0) | null_term);
			} else {
				end += static_cast<idx_t>(code == c) | (static_cast<idx_t>(code == 0) & null_term);
			}
		}
	}
	(void)mask_scan;
	return NumericCast<idx_t>(end - out);
}

//! The pushed filter of a codes-only scan, decided on codes (CodeTranslatable), classified once per scan state from the
//! segment's translation alone: a local code stands for its class - NULL (0), the empty string (the one local code its
//! translation maps to the column's empty code), any other - so the survivors take the single-code path over the local
//! codes, the empty string's local code the one code (a code no row holds where the empty string's verdict is the
//! others'). A filter not decided on codes never reaches a codes-only column: it is refused here
static void ClassifyCodesOnlyFilter(CompressedStringScanState &scan_state, const TableFilter &filter) {
	if (!dict_global::CodeTranslatable(filter)) {
		throw InternalException("dict_global: a filter that is not decided on codes reached a codes-only column");
	}
	auto &translation = *scan_state.global_translation;
	const auto empty_code = translation.dict->persisted->EmptyCode();
	bool null_passes, empty_passes, other_passes;
	dict_global::CodeClassVerdicts(filter, empty_code, null_passes, empty_passes, other_passes);
	auto one_code = UnsafeNumericCast<uint32_t>(scan_state.dict_count);
	if (empty_code != 0 && empty_passes != other_passes) {
		for (idx_t local = 1; local < scan_state.dict_count; local++) {
			if (translation.codes[local] == empty_code) {
				one_code = UnsafeNumericCast<uint32_t>(local);
				break;
			}
		}
	}
	// the non-NULL codes all take other_passes but the one code: ONE_FAILS when they pass, ONE_PASSES when they fail
	scan_state.single_code_shape = other_passes ? SINGLE_CODE_ONE_FAILS : SINGLE_CODE_ONE_PASSES;
	scan_state.single_code = one_code;
	scan_state.single_code_image_valid = false;
	if (other_passes && one_code < scan_state.dict_count && scan_state.dictionary_indices_width > 0 &&
	    scan_state.dictionary_indices_width <= 32) {
		SingleCodeBuildImage(scan_state);
	}
	scan_state.filter_only_null_passes = null_passes;
}

//! The survivors of a classified codes-only filter over the local codes (written to `new_sel`, their count returned);
//! `local` is set to the local codes of the vector's rows (row i at index i) the survivors were read from
static idx_t CodesOnlySurvivors(CompressedStringScanState &scan_state, idx_t start, idx_t vector_count,
                                const SelectionVector &sel, idx_t sel_count, SelectionVector &new_sel,
                                optional_ptr<const SelectionVector> &local) {
	const bool null_passes = scan_state.filter_only_null_passes;
	const bool whole = !sel.IsSet() && sel_count == vector_count;
	idx_t approved = 0;
	if (SingleCodeBlockLoop(scan_state, whole, vector_count, start)) {
		local = &SingleCodeBlockSelVec(scan_state);
		approved = SingleCodeBlockSurvivors(scan_state, start, null_passes, new_sel);
	} else if (whole) {
		local = &scan_state.GetSelVec(start, vector_count);
		approved = SingleCodeSurvivors(scan_state, *local, vector_count, null_passes, new_sel);
	} else {
		local = &scan_state.GetSelVec(start, vector_count);
		auto &codes = *local;
		const auto c = UnsafeNumericCast<sel_t>(scan_state.single_code);
		const bool fails = scan_state.single_code_shape == SINGLE_CODE_ONE_FAILS;
		for (idx_t i = 0; i < sel_count; i++) {
			const auto row = sel.get_index(i);
			const sel_t code = codes.get_index(row);
			new_sel.set_index(approved, row);
			approved += code == 0 ? null_passes : (fails ? code != c : code == c);
		}
	}
	return approved;
}

//! A codes-only scan of a column read for its pushed filter only (MarkFilterOnlyCodes): every filter decided on codes
//! classifies a local code by its translation alone (ClassifyCodesOnlyFilter). No per-row translation and no vector
//! over the global codes: nothing reads the column's values, so the result is NULL
static void FilterOnlyCodes(CompressedStringScanState &scan_state, idx_t start, idx_t vector_count, Vector &result,
                            SelectionVector &sel, idx_t &sel_count, const TableFilter &filter) {
	if (!scan_state.filter_only_classified) {
		ClassifyCodesOnlyFilter(scan_state, filter);
		scan_state.filter_only_classified = true;
	}
	SelectionVector new_sel(sel_count);
	optional_ptr<const SelectionVector> local;
	const idx_t approved = CodesOnlySurvivors(scan_state, start, vector_count, sel, sel_count, new_sel, local);
	if (approved < vector_count) {
		sel.Initialize(new_sel);
	}
	sel_count = approved;
	result.SetVectorType(VectorType::CONSTANT_VECTOR);
	ConstantVector::SetNull(result, true);
}

//! A codes-only scan of a column read for a key (kCodesOnlyKeyFilterPerSegment): its pushed filter is classified once
//! per scan state from the segment's translation as for a filter-only column, the survivors are selected on the local
//! codes, and only the surviving rows are translated into the vector over the global codes (every other row holds the
//! NULL code, a valid index no consumer reads: the vector stays positional over the vector's rows)
static void KeyFilterCodes(CompressedStringScanState &scan_state, idx_t start, idx_t vector_count, Vector &result,
                           SelectionVector &sel, idx_t &sel_count, const TableFilter &filter) {
	if (!scan_state.key_filter_classified) {
		ClassifyCodesOnlyFilter(scan_state, filter);
		scan_state.key_filter_classified = true;
	}
	SelectionVector new_sel(sel_count);
	optional_ptr<const SelectionVector> local;
	const idx_t approved = CodesOnlySurvivors(scan_state, start, vector_count, sel, sel_count, new_sel, local);
	scan_state.ScanToGlobalDictionarySelected(*local, new_sel, approved, vector_count, result);
	if (approved < vector_count) {
		sel.Initialize(new_sel);
	}
	sel_count = approved;
}

static void DictFSSTFilter(ColumnSegment &segment, ColumnScanState &state, idx_t vector_count, Vector &result,
                           SelectionVector &sel, idx_t &sel_count, const TableFilter &filter,
                           TableFilterState &filter_state) {
	auto &scan_state = state.scan_state->Cast<CompressedStringScanState>();
	auto start = state.GetPositionInSegment();
	if (scan_state.codes_only && scan_state.global_translation->filter_only) {
		FilterOnlyCodes(scan_state, start, vector_count, result, sel, sel_count, filter);
		return;
	}
	if (scan_state.codes_only && kCodesOnlyKeyFilterPerSegment) {
		KeyFilterCodes(scan_state, start, vector_count, result, sel, sel_count, filter);
		return;
	}
	if (scan_state.codes_only) {
		auto &local = scan_state.GetSelVec(start, vector_count);
		scan_state.ScanToGlobalDictionary(local, vector_count, result);
		auto &codes = DictionaryVector::SelVector(result);
		dict_global::FilterCodes(*scan_state.global_translation->dict->persisted, filter, codes.data(), vector_count,
		                         sel, sel_count);
		return;
	}
	// a sparse incoming selection of a whole vector evaluates the filter only for the referenced codes; once a
	// scan state has taken the whole-dictionary path (filter_result computed) or carries domain verdicts for this
	// filter it stays on that path
	if (scan_state.mode != DictFSSTMode::FSST_ONLY && vector_count == STANDARD_VECTOR_SIZE && !scan_state.filter_result &&
	    !(scan_state.domain_valid && scan_state.domain_filter == &filter) && sel_count <= SELECTION_AWARE_MAX_SEL_COUNT) {
		SelectionAwareFilter(scan_state, start, vector_count, result, sel, sel_count, filter, filter_state);
		return;
	}
	if (scan_state.AllowDictionaryScan(vector_count)) {
		auto slot = VerdictSlot(scan_state, filter);
		// only pushdown filters on dictionaries
		if (!scan_state.filter_result) {
			// no filter result yet - apply filter to the dictionary
			// initialize the filter result - setting everything to false
			scan_state.filter_result = make_unsafe_uniq_array<bool>(scan_state.dict_count);

			if (slot && slot->IsComplete() && IsConstantFilterKey(slot->filter_key)) {
				// an `=` / `<>` constant's verdicts from its slot, word by word, and the
				// single-code shape the evaluation below would classify
				const idx_t filter_count = slot->ExpandNonNull(scan_state.filter_result.get());
				if (SingleCodeFastPathEnabled()) {
					ClassifySingleCodeFromVerdicts(scan_state, filter_count);
				}
			} else if (slot && slot->IsComplete()) {
				// every non-NULL code's verdict from an earlier request - the whole-dictionary evaluation skipped
				for (idx_t code = 1; code < scan_state.dict_count; code++) {
					scan_state.filter_result[code] = slot->Get(code) == FilterVerdictSlot::PASS;
				}
			} else {
				// Slot zero represents NULL and is not necessarily referenced by any row.
				SelectionVector dict_sel;
				idx_t filter_count = FilterDictionaryEntries(scan_state, filter, filter_state, dict_sel);

				// now set all matching tuples to true
				for (idx_t i = 0; i < filter_count; i++) {
					auto idx = dict_sel.get_index(i) + 1;
					scan_state.filter_result[idx] = true;
				}
				if (SingleCodeFastPathEnabled()) {
					ClassifySingleCode(scan_state, dict_sel, filter_count);
				}
				if (slot) {
					slot->SetAllNonNull(scan_state.filter_result.get());
				}
			}
		}
		// Till now, we have a filter result for all non-NULL values.
		// When the domain check has evaluated this filter - its dynamic bound included - over the dictionary, select
		// the rows by those verdicts: they are a superset of the rows the consumer of an optional filter can use.
		const bool use_domain = scan_state.domain_valid && scan_state.domain_filter == &filter;
		const bool *entry_pass = use_domain ? scan_state.domain_result.get() : scan_state.filter_result.get();
		const bool single_code = scan_state.single_code_shape != SINGLE_CODE_NONE && !sel.IsSet() &&
		                         sel_count == vector_count &&
		                         (!use_domain || (SingleCodeFilterWithVerdictsEnabled() && DomainKeepsSingleCode(scan_state)));
		const bool block_loop = SingleCodeBlockLoop(scan_state, single_code, vector_count, start);
		auto &dict_sel = block_loop ? SingleCodeBlockSelVec(scan_state) : scan_state.GetSelVec(start, vector_count);
		SelectionVector new_sel(sel_count);
		idx_t approved_tuple_count = 0;
		if (block_loop) {
			// the block loop (the NULL slot's verdict as the single-code path reads it; the
			// domain's under a domain verdict, as the verdict loop reads it)
			if (!use_domain) {
				EnsureNullVerdict(scan_state, filter, filter_state);
			}
			approved_tuple_count = SingleCodeBlockSurvivors(scan_state, start, entry_pass[0], new_sel);
		} else if (single_code) {
			// the first pushed predicate over a whole vector, a single-code verdict set; under a
			// domain verdict of the same shape too, the NULL slot's verdict then read from the domain as the loop does
			if (!use_domain) {
				EnsureNullVerdict(scan_state, filter, filter_state);
			}
			approved_tuple_count = SingleCodeSurvivors(scan_state, dict_sel, vector_count, entry_pass[0], new_sel);
		} else if (!use_domain || scan_state.domain_qualifying > 0) {
			for (idx_t idx = 0; idx < sel_count; idx++) {
				auto row_idx = sel.get_index(idx);
				auto dict_offset = dict_sel.get_index(row_idx);
				// Evaluate NULL only when slot zero is referenced by an actual row.
				if (dict_offset == 0 && !use_domain && !scan_state.null_filter_result_initialized) {
					const auto known = slot ? slot->Get(0) : FilterVerdictSlot::UNKNOWN;
					if (known != FilterVerdictSlot::UNKNOWN) {
						scan_state.filter_result[0] = known == FilterVerdictSlot::PASS;
					} else {
						Vector null_data(scan_state.dictionary->data, /*offset=*/0, /*end=*/1);
						UnifiedVectorFormat null_vdata;
						null_data.ToUnifiedFormat(1, null_vdata);
						SelectionVector null_sel;
						idx_t null_filter_count = 1;
						ColumnSegment::FilterSelection(null_sel, null_data, null_vdata, filter, filter_state, 1,
						                               null_filter_count);
						scan_state.filter_result[0] = null_filter_count == 1;
						if (slot) {
							slot->Set(0, scan_state.filter_result[0]);
						}
					}
					scan_state.null_filter_result_initialized = true;
				}
				// Check filter result for the value at the offset and assign selection vector.
				if (!entry_pass[dict_offset]) {
					// does not pass the filter
					continue;
				}
				new_sel.set_index(approved_tuple_count++, row_idx);
			}
		}
		if (approved_tuple_count < vector_count) {
			sel.Initialize(new_sel);
		}
		sel_count = approved_tuple_count;

		if (scan_state.global_translation) {
			scan_state.ScanToGlobalDictionary(dict_sel, vector_count, result);
			return;
		}
		result.Dictionary(scan_state.dictionary, dict_sel);
		return;
	}
	// fallback: scan + filter
	DictFSSTCompressionStorage::StringScan(segment, state, vector_count, result);

	UnifiedVectorFormat vdata;
	result.ToUnifiedFormat(vector_count, vdata);
	ColumnSegment::FilterSelection(sel, result, vdata, filter, filter_state, vector_count, sel_count);
}

static string DictFSSTModeToString(const DictFSSTMode mode) {
	switch (mode) {
	case DictFSSTMode::DICTIONARY:
		return "DICTIONARY";
	case DictFSSTMode::DICT_FSST:
		return "DICT_FSST";
	case DictFSSTMode::FSST_ONLY:
		return "FSST_ONLY";
	default:
		return "UNKNOWN";
	}
}

//===--------------------------------------------------------------------===//
// GetSegmentInfo
//===--------------------------------------------------------------------===//
static InsertionOrderPreservingMap<string> DictFSSTGetSegmentInfo(QueryContext, ColumnSegment &segment) {
	auto &buffer_manager = BufferManager::GetBufferManager(segment.db);
	auto state = make_uniq<CompressedStringScanState>(segment, buffer_manager.Pin(segment.block));
	state->Initialize(false);

	const auto tuple_count = segment.count.load();

	InsertionOrderPreservingMap<string> result;
	result[DictFSSTModeToString(state->mode)] = StringUtil::Format("%d", tuple_count);
	return result;
}

} // namespace dict_fsst

//===--------------------------------------------------------------------===//
// Get Function
//===--------------------------------------------------------------------===//
CompressionFunction DictFSSTCompressionFun::GetFunction(PhysicalType data_type) {
	auto res = CompressionFunction(
	    CompressionType::COMPRESSION_DICT_FSST, data_type, dict_fsst::DictFSSTCompressionStorage::StringInitAnalyze,
	    dict_fsst::DictFSSTCompressionStorage::StringAnalyze, dict_fsst::DictFSSTCompressionStorage::StringFinalAnalyze,
	    dict_fsst::DictFSSTCompressionStorage::InitCompression, dict_fsst::DictFSSTCompressionStorage::Compress,
	    dict_fsst::DictFSSTCompressionStorage::FinalizeCompress, dict_fsst::DictFSSTCompressionStorage::StringInitScan,
	    dict_fsst::DictFSSTCompressionStorage::StringScan,
	    dict_fsst::DictFSSTCompressionStorage::StringScanPartial<false>,
	    dict_fsst::DictFSSTCompressionStorage::StringFetchRow, UncompressedFunctions::EmptySkip,
	    dict_fsst::DictFSSTInitSegment);
	res.validity = CompressionValidity::NO_VALIDITY_REQUIRED;
	res.select = dict_fsst::DictFSSTSelect;
	res.filter = dict_fsst::DictFSSTFilter;
	res.check_domain = dict_fsst::DictFSSTCheckDomain;
	res.get_segment_info = dict_fsst::DictFSSTGetSegmentInfo;
	res.serialize_state = dict_fsst::DictFSSTSerializeState;
	res.deserialize_state = dict_fsst::DictFSSTDeserializeState;
	res.visit_block_ids = dict_fsst::DictFSSTVisitBlockIds;
	res.init_prefetch = dict_fsst::DictFSSTInitPrefetch;

	return res;
}

bool DictFSSTCompressionFun::TypeIsSupported(const PhysicalType physical_type) {
	return physical_type == PhysicalType::VARCHAR;
}

} // namespace duckdb
