#include "duckdb/storage/compression/dict_fsst/filter_verdict_cache.hpp"

#include "duckdb/function/scalar/string_common.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/storage/checkpoint/string_checkpoint_state.hpp"
#include "duckdb/storage/string_uncompressed.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <cstdlib>
#include <cstring>

namespace duckdb {
namespace dict_fsst {

constexpr const uint8_t FilterVerdictSlot::UNKNOWN;
constexpr const uint8_t FilterVerdictSlot::PASS;
constexpr const uint8_t FilterVerdictSlot::FAIL;
constexpr const idx_t FilterVerdictSlot::MAX_SLOTS;
constexpr const idx_t FilterVerdictSlot::CODES_PER_WORD;

//! the longest needle a canonical filter may carry (a longer one takes the plain path)
static constexpr idx_t MAX_NEEDLE_BYTES = 256;

bool FilterVerdictCacheEnabled() {
	return kFilterVerdictCache;
}

bool ConstantFilterCacheEnabled() {
	return kConstantFilterVerdictCache;
}

//! The first byte of a constant comparison's key (a `contains` key starts with its NOT bit, 0 or
//! 1)
static constexpr char CONSTANT_KEY_TAG = '\x02';

bool IsConstantFilterKey(const string &key) {
	return !key.empty() && key[0] == CONSTANT_KEY_TAG;
}

//! The most bytes of slot weight alive at once (kFilterVerdictCacheCapacityMiB)
static idx_t VerdictCapBytes() {
	return idx_t(kFilterVerdictCacheCapacityMiB) * 1024 * 1024;
}

//! the summed weight of the admitted slots that are alive (reserved by compare-and-swap, given back at destruction)
static atomic<idx_t> admitted_bytes {0};

static bool ReserveWeight(idx_t weight) {
	const auto cap = VerdictCapBytes();
	auto current = admitted_bytes.load(std::memory_order_relaxed);
	do {
		if (weight > cap || current > cap - weight) {
			return false;
		}
	} while (!admitted_bytes.compare_exchange_weak(current, current + weight, std::memory_order_relaxed));
	return true;
}

static void ReleaseWeight(idx_t weight) {
	admitted_bytes.fetch_sub(weight, std::memory_order_relaxed);
}

FilterVerdictSlot::FilterVerdictSlot(string filter_key_p, idx_t dict_count_p, idx_t weight_p, bool *created)
    : filter_key(std::move(filter_key_p)), dict_count(dict_count_p), weight(weight_p), complete(false), pass_count(0) {
	const idx_t word_count = (dict_count + CODES_PER_WORD - 1) / CODES_PER_WORD;
	words = make_unsafe_uniq_array_uninitialized<std::atomic<uint64_t>>(word_count);
	for (idx_t i = 0; i < word_count; i++) {
		words[i].store(0, std::memory_order_relaxed);
	}
	*created = true;
}

FilterVerdictSlot::~FilterVerdictSlot() {
	ReleaseWeight(weight);
}

void FilterVerdictSlot::SetAllNonNull(const bool *pass) {
	const idx_t word_count = (dict_count + CODES_PER_WORD - 1) / CODES_PER_WORD;
	idx_t passing = 0;
	for (idx_t w = 0; w < word_count; w++) {
		uint64_t bits = 0;
		const idx_t first = w * CODES_PER_WORD;
		const idx_t last = MinValue<idx_t>(first + CODES_PER_WORD, dict_count);
		for (idx_t code = MaxValue<idx_t>(first, 1); code < last; code++) {
			bits |= static_cast<uint64_t>(pass[code] ? PASS : FAIL) << (2 * (code - first));
			passing += pass[code] ? 1 : 0;
		}
		if (bits) {
			words[w].fetch_or(bits, std::memory_order_relaxed);
		}
	}
	pass_count.store(passing, std::memory_order_relaxed);
	complete.store(true, std::memory_order_release);
}

idx_t FilterVerdictSlot::ExpandNonNull(bool *pass) const {
	const idx_t word_count = (dict_count + CODES_PER_WORD - 1) / CODES_PER_WORD;
	idx_t passing = 0;
	for (idx_t w = 0; w < word_count; w++) {
		const uint64_t bits = words[w].load(std::memory_order_relaxed);
		const idx_t first = w * CODES_PER_WORD;
		const idx_t last = MinValue<idx_t>(first + CODES_PER_WORD, dict_count);
		for (idx_t code = MaxValue<idx_t>(first, 1); code < last; code++) {
			const bool passes = ((bits >> (2 * (code - first))) & 3) == PASS;
			pass[code] = passes;
			passing += passes ? 1 : 0;
		}
	}
	return passing;
}

//! The canonical key of a pushed filter, or false: an EXPRESSION_FILTER whose expression is the built-in VARCHAR
//! `contains` (its callback is GetStringContains()'s) over bound reference 0 of a VARCHAR without collation and a
//! constant non-NULL VARCHAR needle of at most MAX_NEEDLE_BYTES, optionally under one OPERATOR_NOT. The key is the NOT
//! bit followed by the length-prefixed needle bytes.
bool CanonicalFilterKey(const TableFilter &filter, string &key) {
	if (filter.filter_type != TableFilterType::EXPRESSION_FILTER) {
		return false;
	}
	const Expression *expr = filter.Cast<ExpressionFilter>().expr.get();
	bool negated = false;
	if (expr->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
		auto &op = expr->Cast<BoundOperatorExpression>();
		if (op.children.size() != 1) {
			return false;
		}
		negated = true;
		expr = op.children[0].get();
	}
	if (expr->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION ||
	    expr->return_type.id() != LogicalTypeId::BOOLEAN) {
		return false;
	}
	auto &func = expr->Cast<BoundFunctionExpression>();
	using callback_t = void (*)(DataChunk &, ExpressionState &, Vector &);
	static const callback_t contains_callback = []() {
		auto callback = GetStringContains().GetFunctionCallback();
		auto target = callback.target<callback_t>();
		return target ? *target : nullptr;
	}();
	auto callback = func.function.GetFunctionCallback();
	auto target = callback.target<callback_t>();
	if (!contains_callback || !target || *target != contains_callback || func.bind_info ||
	    func.children.size() != 2) {
		return false;
	}
	auto &column = *func.children[0];
	auto &needle = *func.children[1];
	if (column.GetExpressionClass() != ExpressionClass::BOUND_REF ||
	    column.Cast<BoundReferenceExpression>().index != 0 || column.return_type.id() != LogicalTypeId::VARCHAR ||
	    !StringType::GetCollation(column.return_type).empty()) {
		return false;
	}
	if (needle.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	auto &value = needle.Cast<BoundConstantExpression>().value;
	if (value.IsNull() || value.type().id() != LogicalTypeId::VARCHAR ||
	    !StringType::GetCollation(value.type()).empty()) {
		return false;
	}
	auto &bytes = StringValue::Get(value);
	if (bytes.size() > MAX_NEEDLE_BYTES) {
		return false;
	}
	const auto length = static_cast<uint32_t>(bytes.size());
	key.clear();
	key.push_back(negated ? '\x01' : '\x00');
	key.append(reinterpret_cast<const char *>(&length), sizeof(length));
	key.append(bytes);
	return true;
}

//! The canonical key of a CONSTANT_COMPARISON filter, or false - an `=` or `<>` against a
//! non-NULL VARCHAR constant without collation of at most MAX_NEEDLE_BYTES. The key is CONSTANT_KEY_TAG, the comparison
//! (0 for `=`, 1 for `<>`) and the length-prefixed constant bytes: the verdict of a dictionary entry is
//! ColumnSegment::FilterSelection's binary comparison of the entry with the constant, a function of these alone.
static bool CanonicalConstantKey(const TableFilter &filter, string &key) {
	if (filter.filter_type != TableFilterType::CONSTANT_COMPARISON) {
		return false;
	}
	auto &constant_filter = filter.Cast<ConstantFilter>();
	if (constant_filter.comparison_type != ExpressionType::COMPARE_EQUAL &&
	    constant_filter.comparison_type != ExpressionType::COMPARE_NOTEQUAL) {
		return false;
	}
	auto &value = constant_filter.constant;
	if (value.IsNull() || value.type().id() != LogicalTypeId::VARCHAR ||
	    !StringType::GetCollation(value.type()).empty()) {
		return false;
	}
	auto &bytes = StringValue::Get(value);
	if (bytes.size() > MAX_NEEDLE_BYTES) {
		return false;
	}
	const auto length = static_cast<uint32_t>(bytes.size());
	key.clear();
	key.push_back(CONSTANT_KEY_TAG);
	key.push_back(constant_filter.comparison_type == ExpressionType::COMPARE_EQUAL ? '\x00' : '\x01');
	key.append(reinterpret_cast<const char *>(&length), sizeof(length));
	key.append(bytes);
	return true;
}

static string SlotKey(const string &dictionary_cache_key, idx_t i) {
	return dictionary_cache_key + "#v" + to_string(i);
}

//===--------------------------------------------------------------------===//
// Segment-local references to the segment's ObjectCache entries
//===--------------------------------------------------------------------===//
bool SegmentCacheEnabled() {
	return kSegmentObjectCacheReferences;
}

//! A DICT_FSST segment's state - UncompressedStringStorage's, as before - plus weak references to the entries found or
//! published under the segment's current dictionary cache key, all under `lock`. The key is the segment's own (a fresh
//! UUID per binding) and EnsureDictionary is the only publisher under it, so `published` false (it is set before a
//! publication begins) means the ObjectCache holds no dictionary entry under the key. A reference is taken only from an
//! entry the ObjectCache returned under the current key, and every key replacement clears them (ResetSegmentCache). A
//! weak reference keeps no entry alive (an evicted entry is destroyed and its reservation released as before; only its
//! control block's allocation stays until the reference is replaced).
struct DictFSSTSegmentState : public UncompressedStringSegmentState {
	mutex lock;
	bool published = false;
	weak_ptr<ObjectCacheEntry> dictionary;
	weak_ptr<FilterVerdictSlot> slots[FilterVerdictSlot::MAX_SLOTS];
	//! the segment translation answer under the current key (`translation` expired or empty for an
	//! absence) and the translation generation it was taken at; valid only while `translation_known`
	bool translation_known = false;
	bool translation_present = false;
	idx_t translation_generation = 0;
	weak_ptr<ObjectCacheEntry> translation;
};

static optional_ptr<DictFSSTSegmentState> SegmentCache(ColumnSegment &segment) {
	if (!SegmentCacheEnabled()) {
		return nullptr;
	}
	auto state = segment.GetSegmentState();
	return state ? dynamic_cast<DictFSSTSegmentState *>(state.get()) : nullptr;
}

unique_ptr<CompressedSegmentState> DictFSSTInitSegment(ColumnSegment &segment, block_id_t block_id,
                                                       optional_ptr<ColumnSegmentState> segment_state) {
	auto result = UncompressedStringStorage::StringInitSegment(segment, block_id, segment_state);
	if (!SegmentCacheEnabled()) {
		return result;
	}
	// the standard state holds nothing but the deserialized overflow block ids at this point
	auto state = make_uniq<DictFSSTSegmentState>();
	state->on_disk_blocks = std::move(result->Cast<UncompressedStringSegmentState>().on_disk_blocks);
	return std::move(state);
}

SegmentDictionaryLookup LookupSegmentDictionary(ColumnSegment &segment, shared_ptr<ObjectCacheEntry> &entry) {
	auto state = SegmentCache(segment);
	if (!state) {
		return SegmentDictionaryLookup::ASK_CACHE;
	}
	lock_guard<mutex> guard(state->lock);
	entry = state->dictionary.lock();
	if (entry) {
		return SegmentDictionaryLookup::HIT;
	}
	return state->published ? SegmentDictionaryLookup::ASK_CACHE : SegmentDictionaryLookup::ABSENT;
}

void NoteSegmentDictionaryPublication(ColumnSegment &segment) {
	auto state = SegmentCache(segment);
	if (state) {
		lock_guard<mutex> guard(state->lock);
		state->published = true;
	}
}

void RememberSegmentDictionary(ColumnSegment &segment, const shared_ptr<ObjectCacheEntry> &entry) {
	auto state = SegmentCache(segment);
	if (state && entry) {
		lock_guard<mutex> guard(state->lock);
		state->published = true;
		state->dictionary = entry;
	}
}

void ResetSegmentCache(ColumnSegment &segment) {
	auto state = SegmentCache(segment);
	if (state) {
		lock_guard<mutex> guard(state->lock);
		state->published = false;
		state->dictionary.reset();
		for (auto &slot : state->slots) {
			slot.reset();
		}
		state->translation_known = false;
		state->translation.reset();
	}
}

bool LookupSegmentTranslation(ColumnSegment &segment, idx_t generation, shared_ptr<ObjectCacheEntry> &entry) {
	auto state = SegmentCache(segment);
	if (!state) {
		return false;
	}
	lock_guard<mutex> guard(state->lock);
	if (!state->translation_known || state->translation_generation != generation) {
		return false;
	}
	if (!state->translation_present) {
		entry = nullptr;
		return true;
	}
	entry = state->translation.lock();
	return entry != nullptr;
}

void RememberSegmentTranslation(ColumnSegment &segment, idx_t generation, const shared_ptr<ObjectCacheEntry> &entry) {
	auto state = SegmentCache(segment);
	if (state) {
		lock_guard<mutex> guard(state->lock);
		state->translation_known = true;
		state->translation_present = entry != nullptr;
		state->translation_generation = generation;
		state->translation = entry;
	}
}

shared_ptr<FilterVerdictSlot> AcquireFilterVerdictSlot(ColumnSegment &segment, idx_t dict_count,
                                                       const TableFilter &filter) {
	if (!FilterVerdictCacheEnabled() || segment.segment_type != ColumnSegmentType::PERSISTENT || dict_count == 0) {
		return nullptr;
	}
	auto &dictionary_cache_key = segment.GetDictionaryCacheKey();
	if (dictionary_cache_key.empty()) {
		return nullptr;
	}
	string filter_key;
	if (!CanonicalFilterKey(filter, filter_key) &&
	    !(ConstantFilterCacheEnabled() && CanonicalConstantKey(filter, filter_key))) {
		return nullptr;
	}
	auto segment_cache = SegmentCache(segment);
	if (segment_cache) {
		lock_guard<mutex> guard(segment_cache->lock);
		for (auto &remembered : segment_cache->slots) {
			auto slot = remembered.lock();
			if (slot && slot->filter_key == filter_key) {
				return slot->dict_count == dict_count ? slot : nullptr;
			}
		}
	}
	auto &cache = segment.db.GetObjectCache();
	for (idx_t i = 0; i < FilterVerdictSlot::MAX_SLOTS; i++) {
		auto slot_key = SlotKey(dictionary_cache_key, i);
		auto slot = cache.Get<FilterVerdictSlot>(slot_key);
		if (!slot) {
			// the first missing index: create the slot there, its weight reserved on the cap first
			const idx_t word_count =
			    (dict_count + FilterVerdictSlot::CODES_PER_WORD - 1) / FilterVerdictSlot::CODES_PER_WORD;
			const idx_t weight =
			    sizeof(FilterVerdictSlot) + filter_key.size() + slot_key.size() + word_count * sizeof(uint64_t);
			if (!ReserveWeight(weight)) {
				return nullptr;
			}
			bool created = false;
			slot = cache.GetOrCreate<FilterVerdictSlot>(slot_key, filter_key, dict_count, weight, &created);
			if (!created) {
				// a racing creator inserted this index first (or the key holds another type): give the reservation back
				ReleaseWeight(weight);
			}
			if (!slot) {
				return nullptr;
			}
		}
		if (segment_cache) {
			lock_guard<mutex> guard(segment_cache->lock);
			segment_cache->slots[i] = slot;
		}
		if (slot->filter_key == filter_key) {
			if (slot->dict_count != dict_count) {
				return nullptr;
			}
			return slot;
		}
	}
	// a fifth filter on this dictionary: not admitted
	return nullptr;
}

void DeleteFilterVerdictSlots(ObjectCache &cache, const string &dictionary_cache_key) {
	for (idx_t i = 0; i < FilterVerdictSlot::MAX_SLOTS; i++) {
		cache.Delete(SlotKey(dictionary_cache_key, i));
	}
}

} // namespace dict_fsst
} // namespace duckdb
