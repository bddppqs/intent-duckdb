#pragma once

#include "duckdb/storage/compression/dict_fsst/common.hpp"
#include "duckdb/storage/compression/dict_fsst/filter_verdict_cache.hpp"
#include "duckdb/planner/table_filter_state.hpp"

namespace duckdb {
class TableFilter;
struct ColumnScanState;
namespace dict_global {
class SegmentTranslation;
}

namespace dict_fsst {

//===--------------------------------------------------------------------===//
// Scan
//===--------------------------------------------------------------------===//
struct CompressedStringScanState : public SegmentScanState {
public:
	CompressedStringScanState(ColumnSegment &segment, BufferHandle &&handle_p)
	    : segment(segment), owned_handle(std::move(handle_p)), handle(owned_handle) {
	}
	CompressedStringScanState(ColumnSegment &segment, BufferHandle &handle_p)
	    : segment(segment), owned_handle(), handle(handle_p) {
	}

	~CompressedStringScanState() override;

public:
	void Initialize(bool initialize_dictionary = true);
	//! Materialise the whole dictionary on demand (idempotent): a no-op for a fetch state, an FSST_ONLY segment or a
	//! cache hit; publishes to the dictionary cache exactly as a non-deferred Initialize(true) does
	void EnsureDictionary();
	//! Decode entry `code` (1 <= code < dict_count) once into the scan-state string buffer
	string_t DecodeEntry(idx_t code);
	//! Decode the entries 1..dict_count-1 into `target` (entry i at row i - 1) without touching the on-demand
	//! decode state below; DICT_FSST entries land in `target`'s string heap, DICTIONARY-mode entries view the pinned block
	void DecodeEntriesInto(Vector &target);
	//! Whether a partial piece of this segment should decode its referenced codes on demand instead of materialising:
	//! the segment is already in the sparse mode, or it is reached inside a vector after a segment that stayed sparse
	bool PreferOnDemandDecode(optional_ptr<ColumnScanState> column_state) const;
	void ScanToFlatVector(Vector &result, idx_t result_offset, idx_t start, idx_t scan_count,
	                      optional_ptr<ColumnScanState> column_state = nullptr);
	void ScanToDictionaryVector(ColumnSegment &segment, Vector &result, idx_t result_offset, idx_t start,
	                            idx_t scan_count);
	const SelectionVector &GetSelVec(idx_t start, idx_t scan_count);
	void Select(Vector &result, idx_t start, const SelectionVector &sel, idx_t sel_count);

	bool AllowDictionaryScan(idx_t scan_count);
	//! Whether a whole vector of this segment is emitted over the published global dictionary
	bool AllowGlobalDictionaryScan(idx_t scan_count) const;
	//! Emit `count` rows whose local codes are `local` over the published global dictionary
	void ScanToGlobalDictionary(const SelectionVector &local, idx_t count, Vector &result);

	//! Entry `dict_idx` of a state initialised without its dictionary, at `dict_offset` (the sum of the entries'
	//! string_lengths before it): decoded into `result`'s string heap (DICT_FSST) or a view into the pinned block
	//! (DICTIONARY) - the global dictionary's parallel build
	string_t FetchEntry(Vector &result, uint32_t dict_offset, idx_t dict_idx) {
		return FetchStringFromDict(result, dict_offset, dict_idx);
	}

private:
	string_t FetchStringFromDict(Vector &result, uint32_t dict_offset, idx_t dict_idx);

public:
	ColumnSegment &segment;
	BufferHandle owned_handle;
	optional_ptr<BufferHandle> handle;

	DictFSSTMode mode;
	idx_t dictionary_size;
	uint32_t dict_count;
	bitpacking_width_t dictionary_indices_width;
	bitpacking_width_t string_lengths_width;

	buffer_ptr<SelectionVector> sel_vec;
	idx_t sel_vec_size = 0;

	// decompress offset/position - used for scanning without a dictionary
	uint32_t decompress_offset = 0;
	idx_t decompress_position = 0;

	vector<uint32_t> string_lengths;

	//! Start of the block (pointing to the dictionary_header)
	data_ptr_t baseptr;
	data_ptr_t dict_ptr;
	data_ptr_t dictionary_indices_ptr;
	data_ptr_t string_lengths_ptr;

	buffer_ptr<VectorChildBuffer> dictionary;
	void *decoder = nullptr;
	bool all_values_inlined = false;

	unsafe_unique_array<bool> filter_result;
	bool null_filter_result_initialized = false;
	//! the shape of filter_result over entries 1..dict_count-1, set when filter_result is computed - 0 none,
	//! 1 exactly one failing entry (a <> whose constant is in the dictionary), 2 exactly one passing entry (an =) -
	//! and that one entry's code
	uint8_t single_code_shape = 0;
	uint32_t single_code = 0;
	//! the packed image of 32 copies of the one failing code at dictionary_indices_width (width x 4 bytes),
	//! built when the ONE_FAILS shape is classified
	uint32_t single_code_image[32];
	bool single_code_image_valid = false;

	//! deferred materialisation: Initialize(true) ran on a DICT_FSST/DICTIONARY segment without a cache hit and
	//! left the whole-dictionary decode to EnsureDictionary(); the admission decided at Initialize is kept for the
	//! publication; decoded_on_demand marks the sparse mode (entries decoded without materialising)
	bool dictionary_deferred = false;
	bool cache_dictionary_admitted = false;
	bool decoded_on_demand = false;
	//! entry offsets (a one-time prefix sum of string_lengths, dict_count + 1 entries) for the on-demand decode
	vector<uint32_t> entry_offsets;
	//! the on-demand decoded strings live in this vector's string heap (referenced from every result that uses them)
	unique_ptr<Vector> decode_buffer;
	//! per dictionary code: the decoded entry and whether it is decoded; the selection-aware filter's verdict
	//! (0 unknown, 1 pass, 2 fail, 3 pending inside one call)
	unsafe_unique_array<string_t> decoded;
	unsafe_unique_array<uint8_t> decoded_known;
	unsafe_unique_array<uint8_t> verdicts;
	//! scratch of the selection-aware filter: the codes gathered for one batch
	vector<idx_t> pending_codes;
	//! the cross-request verdict slot of the pushed filter (nullptr: cache off, filter not canonical or slot not
	//! admitted), resolved once for the filter it was resolved for
	shared_ptr<FilterVerdictSlot> verdict_slot;
	const TableFilter *verdict_filter = nullptr;

	//! Dictionary-domain verdicts of the pushed filter - its constant comparisons AND the dynamic bounds seen at the
	//! last check - for every entry including slot zero, the filter they belong to, the bound key they were computed
	//! under and the number of qualifying entries; domain_unsupported marks a filter shape the check cannot evaluate
	unsafe_unique_array<bool> domain_result;
	idx_t domain_qualifying = 0;
	bool domain_valid = false;
	bool domain_unsupported = false;
	const TableFilter *domain_filter = nullptr;
	string domain_key;
	string domain_probe;

	//! the segment's translation (local code -> global code) when its column is published, else null; and the
	//! translated selection of the last vector
	shared_ptr<dict_global::SegmentTranslation> global_translation;
	buffer_ptr<SelectionVector> global_dictionary_sel;
	idx_t global_dictionary_sel_size = 0;

	//! the filter last resolved and its canonical key (empty: not canonical); the filter state of the eager
	//! whole-dictionary evaluation, built once per scan state for the filter it was built for
	const TableFilter *segment_skip_filter = nullptr;
	string segment_skip_key;
	const TableFilter *segment_skip_eval_filter = nullptr;
	unique_ptr<TableFilterState> segment_skip_eval_state;
};

//! The pushed filter of a vector that ColumnData::Filter scanned flat across segment pieces
//! (piece p: rows [sum of the earlier counts, + piece_counts[p]) of `result`, starting at row piece_starts[p] of its segment,
//! scanned by piece_states[p], nullptr when that segment is not DICT_FSST). A selected row of a DICT_FSST piece whose segment
//! holds a verdict slot for `filter` takes its code's slot verdict; every other selected row is evaluated by
//! ColumnSegment::FilterSelection as before and its verdict published. Returns false (nothing changed) when it declines.
bool StraddleVerdictFilter(SegmentScanState *const *piece_states, const idx_t *piece_starts, const idx_t *piece_counts,
                           idx_t piece_count, Vector &result, UnifiedVectorFormat &vdata, SelectionVector &sel,
                           idx_t scan_count, idx_t &sel_count, const TableFilter &filter, TableFilterState &filter_state);

} // namespace dict_fsst

} // namespace duckdb
