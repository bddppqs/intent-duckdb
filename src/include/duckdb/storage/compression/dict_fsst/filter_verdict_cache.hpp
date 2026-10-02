//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/compression/dict_fsst/filter_verdict_cache.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/storage/object_cache.hpp"

namespace duckdb {
class ColumnSegment;
class TableFilter;
struct ColumnSegmentState;
struct CompressedSegmentState;

namespace dict_fsst {

//! The per-code verdicts of one canonical `contains` filter over one DICT_FSST dictionary, kept in the object
//! cache across requests under `<dictionary cache key>#v<i>` (i = 0..3). A slot holds two bits per dictionary code
//! (00 unknown, 01 pass, 10 fail; code 0 is the NULL entry) and a `complete` flag that is set once every non-NULL code
//! is known. It holds no count, selection or per-row answer: every scan still tests every selected row's code.
class FilterVerdictSlot : public ObjectCacheEntry {
public:
	static constexpr uint8_t UNKNOWN = 0;
	static constexpr uint8_t PASS = 1;
	static constexpr uint8_t FAIL = 2;
	//! the most slots, and so the most distinct canonical filters, per dictionary
	static constexpr idx_t MAX_SLOTS = 4;
	static constexpr idx_t CODES_PER_WORD = 32;

	//! `created` is set by the constructor only: GetOrCreate constructs an entry only for the caller that inserts it,
	//! so a caller that finds it still false got another creator's entry and gives its cap reservation back
	FilterVerdictSlot(string filter_key, idx_t dict_count, idx_t weight, bool *created);
	~FilterVerdictSlot() override;

	static string ObjectType() {
		return "dict_fsst_filter_verdicts";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		return weight;
	}

	uint8_t Get(idx_t code) const {
		auto word = words[code / CODES_PER_WORD].load(std::memory_order_relaxed);
		return static_cast<uint8_t>((word >> (2 * (code % CODES_PER_WORD))) & 3);
	}
	//! Writes only set bits, so racing writers of the same (pure) verdict agree
	void Set(idx_t code, bool pass) {
		auto bits = static_cast<uint64_t>(pass ? PASS : FAIL) << (2 * (code % CODES_PER_WORD));
		words[code / CODES_PER_WORD].fetch_or(bits, std::memory_order_relaxed);
	}
	//! Every non-NULL code 1..dict_count-1 from a full evaluation: `pass[i]` is the verdict of code i
	void SetAllNonNull(const bool *pass);
	//! Write every non-NULL code's verdict of a complete slot into `pass[1..dict_count-1]` (one
	//! word load per 32 codes; `pass[0]` untouched) and return the number of passing codes
	idx_t ExpandNonNull(bool *pass) const;
	bool IsComplete() const {
		return complete.load(std::memory_order_acquire);
	}
	//! The number of passing non-NULL codes, set by SetAllNonNull before the slot is marked complete (read it
	//! only after IsComplete() returned true)
	idx_t PassCount() const {
		return pass_count.load(std::memory_order_relaxed);
	}

	const string filter_key;
	const idx_t dict_count;
	const idx_t weight;

private:
	unsafe_unique_array<std::atomic<uint64_t>> words;
	std::atomic<bool> complete;
	std::atomic<idx_t> pass_count;
};

//! Whether the verdict cache is enabled (a compile-time constant)
bool FilterVerdictCacheEnabled();
//! Whether a pushed `=` / `<>` comparison with a
//! non-NULL VARCHAR constant (ConstantFilter) is admitted to the verdict cache as well, under a key tagged apart from
//! the `contains` keys, so its whole-dictionary evaluation (the dense filter path and the domain check) runs once per
//! dictionary; off, only `contains` / NOT `contains` filters hold slots
bool ConstantFilterCacheEnabled();
//! Whether a slot key is a constant comparison's (never a `contains` key)
bool IsConstantFilterKey(const string &key);
//! The slot for the pushed filter on this segment's dictionary: found, or created when the filter is canonical, the
//! segment is a persistent DICT_FSST segment with a dictionary cache key, fewer than MAX_SLOTS other filters hold slots
//! and the slot-weight capacity admits its weight; nullptr otherwise (the caller's plain path)
shared_ptr<FilterVerdictSlot> AcquireFilterVerdictSlot(ColumnSegment &segment, idx_t dict_count,
                                                       const TableFilter &filter);
//! The removal hook: delete the slots `<key>#v0`..`<key>#v3` of a dictionary cache key that is being retired
void DeleteFilterVerdictSlots(ObjectCache &cache, const string &dictionary_cache_key);
//! The canonical key of a pushed filter (the NOT bit, the needle length, the needle bytes), or false when the
//! filter is not a canonical `contains` / NOT `contains` filter - the admission test of AcquireFilterVerdictSlot
bool CanonicalFilterKey(const TableFilter &filter, string &key);

//! Whether the segment-local references are enabled. Enabled, a DICT_FSST segment's state keeps weak references to the
//! dictionary entry and the verdict slots found or published under its current dictionary cache key, so a later scan
//! resolves them under the segment's own mutex instead of the ObjectCache's process-wide lock; off, every lookup goes
//! to the ObjectCache (the standard segment state)
bool SegmentCacheEnabled();
//! DICT_FSST's init_segment - UncompressedStringStorage::StringInitSegment, its state moved into the segment-local
//! subclass when enabled
unique_ptr<CompressedSegmentState> DictFSSTInitSegment(ColumnSegment &segment, block_id_t block_id,
                                                       optional_ptr<ColumnSegmentState> segment_state);
//! A segment-local dictionary lookup under the segment's current key - HIT with `entry` set; ABSENT when no
//! publication under that key has begun, so the ObjectCache holds no entry under it; ASK_CACHE otherwise
enum class SegmentDictionaryLookup : uint8_t { ASK_CACHE, HIT, ABSENT };
SegmentDictionaryLookup LookupSegmentDictionary(ColumnSegment &segment, shared_ptr<ObjectCacheEntry> &entry);
//! Called before a publication (ObjectCache::GetOrCreate) under the segment's current key begins
void NoteSegmentDictionaryPublication(ColumnSegment &segment);
//! Remember a dictionary entry the ObjectCache returned under the segment's current key (a null entry is ignored)
void RememberSegmentDictionary(ColumnSegment &segment, const shared_ptr<ObjectCacheEntry> &entry);
//! Forget every reference and publication mark; called after the segment's dictionary cache key is replaced
void ResetSegmentCache(ColumnSegment &segment);
//! The segment translation answer remembered in the segment-local state - true with `entry` set to
//! the remembered entry (null for a remembered absence; an expired entry answers false) when it was taken at
//! `generation`; false otherwise, and always when the segment-local references are off (the ObjectCache is asked)
bool LookupSegmentTranslation(ColumnSegment &segment, idx_t generation, shared_ptr<ObjectCacheEntry> &entry);
//! Remember the ObjectCache's translation answer (an entry or null) for the segment's current key,
//! taken at `generation`, as a weak reference
void RememberSegmentTranslation(ColumnSegment &segment, idx_t generation, const shared_ptr<ObjectCacheEntry> &entry);

} // namespace dict_fsst
} // namespace duckdb
