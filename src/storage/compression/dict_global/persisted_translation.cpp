#include "duckdb/storage/compression/dict_global/persisted_translation.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/bitpacking.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/fsst.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/tuning_defaults.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/null_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/compression/dict_fsst/common.hpp"
#include "duckdb/storage/compression/dict_fsst/decompression.hpp"
#include "duckdb/storage/compression/dict_fsst/split_segment.hpp"
#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"
#include "duckdb/storage/table/column_data.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "fsst.h"

#include <algorithm>

namespace duckdb {
namespace dict_global {

bool PersistedTranslationsEnabled() {
	return kPersistedTranslations && kSplitDictionarySegments;
}

void PersistedColumn::Serialize(Serializer &serializer) const {
	serializer.WriteProperty<idx_t>(100, "storage_index", storage_index);
	serializer.WriteProperty<uint64_t>(101, "identity", identity);
	serializer.WriteProperty<idx_t>(102, "count", count);
	serializer.WriteProperty<idx_t>(103, "segments", segments);
	serializer.WriteProperty<idx_t>(104, "rows", rows);
	serializer.WriteProperty<idx_t>(105, "blob_bytes", blob_bytes);
	serializer.WriteProperty<vector<int64_t>>(106, "blocks", blocks);
}

PersistedColumn PersistedColumn::Deserialize(Deserializer &deserializer) {
	PersistedColumn result;
	result.storage_index = deserializer.ReadProperty<idx_t>(100, "storage_index");
	result.identity = deserializer.ReadProperty<uint64_t>(101, "identity");
	result.count = deserializer.ReadProperty<idx_t>(102, "count");
	result.segments = deserializer.ReadProperty<idx_t>(103, "segments");
	result.rows = deserializer.ReadProperty<idx_t>(104, "rows");
	result.blob_bytes = deserializer.ReadProperty<idx_t>(105, "blob_bytes");
	result.blocks = deserializer.ReadProperty<vector<int64_t>>(106, "blocks");
	return result;
}

//===--------------------------------------------------------------------===//
// The blob
//===--------------------------------------------------------------------===//
// A stream over the blob's blocks in which no record crosses a block boundary (a record that does not fit the rest of
// a block starts the next one): a 64-byte header, one 64-byte directory entry per segment in collection order, then each
// segment's data - its new-entry bitmap (one bit per local entry, 64-bit words) and its old entries' codes bit-packed at
// the entry's width in groups of 32.
namespace {
static constexpr uint64_t BLOB_MAGIC = 0x3154414C58525453ULL; // "STRXLAT1"
static constexpr uint32_t BLOB_VERSION = 1;
struct BlobHeader {
	uint64_t magic;
	uint32_t version;
	uint32_t empty_code;
	uint64_t storage_index;
	uint64_t count;
	uint64_t segments;
	uint64_t rows;
	uint64_t reserved[2];
};
struct BlobEntry {
	int64_t block_id;
	uint32_t offset;
	uint32_t dict_count;
	uint32_t start_code;
	uint32_t new_count;
	uint64_t data_offset;
	uint64_t rows;
	uint8_t old_width;
	uint8_t pad[7];
	uint64_t reserved;
};
static_assert(sizeof(BlobHeader) == 64, "blob header layout");
static_assert(sizeof(BlobEntry) == 48 + 8, "blob entry layout");
static constexpr idx_t ENTRY_SIZE = 64;

//! The position of a record of `bytes` at or after `position` that does not cross a block boundary
idx_t Place(idx_t position, idx_t bytes, idx_t block_size) {
	position = AlignValue<idx_t>(position);
	if (position % block_size + bytes > block_size) {
		position = (position / block_size + 1) * block_size;
	}
	return position;
}

idx_t BitmapBytes(idx_t dict_count) {
	return (dict_count + 63) / 64 * sizeof(uint64_t);
}

idx_t OldBytes(idx_t old_count, uint8_t width) {
	return width == 0 ? 0 : BitpackingPrimitives::GetRequiredSize(old_count, width);
}

uint8_t WidthOf(uint32_t max_value) {
	return max_value == 0 ? 0 : BitpackingPrimitives::MinimumBitWidth<uint32_t, false>(max_value);
}
} // namespace

PersistedTranslations::PersistedTranslations(DatabaseInstance &db_p, BlockManager &block_manager_p,
                                             PersistedColumn column_p)
    : db(db_p), block_manager(block_manager_p), column(std::move(column_p)), storage_index(column.storage_index),
      loaded(false) {
}

PersistedTranslations::~PersistedTranslations() {
}

const_data_ptr_t PersistedTranslations::Read(idx_t offset, idx_t bytes, BufferHandle &handle) {
	const idx_t block_size = block_manager.GetBlockSize();
	const idx_t block_index = offset / block_size;
	if (block_index >= column.blocks.size() || offset % block_size + bytes > block_size) {
		throw IOException("Stored column translations: a read outside the blob - the database file appears corrupted");
	}
	if (handles.size() != column.blocks.size()) {
		throw InternalException("Stored column translations read before their blocks were registered");
	}
	handle = BufferManager::GetBufferManager(db).Pin(handles[block_index]);
	return handle.Ptr() + offset % block_size;
}

void PersistedTranslations::EnsureDirectory() {
	if (loaded.load(std::memory_order_acquire)) {
		return;
	}
	lock_guard<mutex> guard(lock);
	if (loaded.load(std::memory_order_relaxed)) {
		return;
	}
	handles.clear();
	for (auto block_id : column.blocks) {
		handles.push_back(block_manager.RegisterBlock(block_id));
	}
	const idx_t block_size = block_manager.GetBlockSize();
	BufferHandle handle;
	BlobHeader header;
	memcpy(&header, Read(0, sizeof(BlobHeader), handle), sizeof(BlobHeader));
	if (header.magic != BLOB_MAGIC || header.version != BLOB_VERSION || header.storage_index != storage_index ||
	    header.count != column.count || header.segments != column.segments) {
		throw IOException("Stored column translations: the blob header does not match its table entry - the database "
		                  "file appears corrupted");
	}
	empty_code = header.empty_code;
	entries.clear();
	entries.reserve(header.segments);
	by_location.clear();
	inverse_starts.clear();
	inverse_entries.clear();
	idx_t position = sizeof(BlobHeader);
	for (idx_t k = 0; k < header.segments; k++) {
		position = Place(position, ENTRY_SIZE, block_size);
		BlobEntry blob_entry;
		memcpy(&blob_entry, Read(position, sizeof(BlobEntry), handle), sizeof(BlobEntry));
		position += ENTRY_SIZE;
		Entry entry;
		entry.block_id = blob_entry.block_id;
		entry.offset = blob_entry.offset;
		entry.dict_count = blob_entry.dict_count;
		entry.start_code = blob_entry.start_code;
		entry.new_count = blob_entry.new_count;
		entry.data_offset = blob_entry.data_offset;
		entry.rows = blob_entry.rows;
		entry.old_width = blob_entry.old_width;
		by_location[(static_cast<uint64_t>(entry.block_id) << 20) ^ entry.offset] = entries.size();
		if (entry.new_count > 0) {
			inverse_starts.push_back(entry.start_code);
			inverse_entries.push_back(NumericCast<uint32_t>(entries.size()));
		}
		entries.push_back(entry);
	}
	loaded.store(true, std::memory_order_release);
}

const vector<PersistedTranslations::Entry> &PersistedTranslations::Entries() {
	EnsureDirectory();
	return entries;
}

uint32_t PersistedTranslations::EmptyCode() {
	EnsureDirectory();
	return empty_code;
}

idx_t PersistedTranslations::FindEntry(int64_t block_id, uint32_t offset) {
	EnsureDirectory();
	auto found = by_location.find((static_cast<uint64_t>(block_id) << 20) ^ offset);
	if (found == by_location.end()) {
		return DConstants::INVALID_INDEX;
	}
	auto &entry = entries[found->second];
	return entry.block_id == block_id && entry.offset == offset ? found->second : DConstants::INVALID_INDEX;
}

void PersistedTranslations::Decode(idx_t entry_index, uint32_t *codes) {
	EnsureDirectory();
	auto &entry = entries[entry_index];
	const idx_t dict_count = entry.dict_count;
	const idx_t old_count = dict_count - 1 - entry.new_count;
	const idx_t bitmap_bytes = BitmapBytes(dict_count);
	BufferHandle handle;
	auto data = Read(entry.data_offset, bitmap_bytes + OldBytes(old_count, entry.old_width), handle);
	auto words = reinterpret_cast<const uint64_t *>(data);
	unsafe_unique_array<uint32_t> old;
	if (old_count > 0) {
		old = make_unsafe_uniq_array_uninitialized<uint32_t>(
		    BitpackingPrimitives::RoundUpToAlgorithmGroupSize<idx_t>(old_count));
		BitpackingPrimitives::UnPackBuffer<uint32_t>(data_ptr_cast(old.get()), const_cast<data_ptr_t>(data + bitmap_bytes),
		                                             old_count, entry.old_width);
	}
	// the new entries take the next codes in local order, the others the old codes in order (branch-free: a segment's
	// new and old entries interleave without pattern)
	if (dict_count - 1 != entry.new_count + old_count) {
		throw IOException("Stored column translations: a segment entry is inconsistent - the database file appears "
		                  "corrupted");
	}
	uint32_t pad = 0;
	const uint32_t *old_codes = old_count > 0 ? old.get() : &pad;
	codes[0] = 0;
	uint32_t next = entry.start_code;
	idx_t next_old = 0;
	for (idx_t local = 1; local < dict_count; local++) {
		const uint32_t bit = uint32_t((words[local / 64] >> (local % 64)) & 1);
		const uint32_t old_code = old_codes[MinValue<idx_t>(next_old, old_count == 0 ? 0 : old_count - 1)];
		codes[local] = bit ? next : old_code;
		next += bit;
		next_old += 1 - bit;
	}
}

string_t PersistedTranslations::Fetch(Vector &result, uint32_t code) {
	EnsureDirectory();
	if (code == 0 || code >= column.count || inverse_starts.empty()) {
		throw InternalException("Stored column translations: code %llu outside the code space", (unsigned long long)code);
	}
	// the segment where the code occurs first, and its rank among that segment's new entries
	auto it = std::upper_bound(inverse_starts.begin(), inverse_starts.end(), code);
	if (it == inverse_starts.begin()) {
		throw InternalException("Stored column translations: code %llu below the first start", (unsigned long long)code);
	}
	const idx_t position = NumericCast<idx_t>(it - inverse_starts.begin()) - 1;
	auto &entry = entries[inverse_entries[position]];
	idx_t rank = code - entry.start_code;
	if (rank >= entry.new_count) {
		throw InternalException("Stored column translations: code %llu past its segment's new entries",
		                        (unsigned long long)code);
	}
	idx_t local = 0;
	{
		BufferHandle handle;
		auto words = reinterpret_cast<const uint64_t *>(Read(entry.data_offset, BitmapBytes(entry.dict_count), handle));
		for (idx_t w = 0; w * 64 < entry.dict_count; w++) {
			const auto ones = idx_t(__builtin_popcountll(words[w]));
			if (rank < ones) {
				uint64_t word = words[w];
				for (idx_t r = 0; r < rank; r++) {
					word &= word - 1;
				}
				local = w * 64 + idx_t(__builtin_ctzll(word));
				break;
			}
			rank -= ones;
		}
	}
	// the string of local entry `local` of the segment's dictionary, read from the segment's own block
	auto &buffer_manager = BufferManager::GetBufferManager(db);
	shared_ptr<BlockHandle> block;
	{
		lock_guard<mutex> guard(lock);
		auto &held = fetch_blocks[entry.block_id];
		if (!held) {
			held = block_manager.RegisterBlock(entry.block_id);
		}
		block = held;
	}
	auto handle = buffer_manager.Pin(block);
	auto base = handle.Ptr() + entry.offset;
	dict_fsst::dict_fsst_compression_header_t header;
	memcpy(&header, base, sizeof(header));
	if (local == 0 || local >= header.dict_count) {
		throw IOException("Stored column translations: a code's segment does not hold it - the database file appears "
		                  "corrupted");
	}
	auto dictionary_dest = AlignValue<idx_t>(dict_fsst::DictFSSTCompression::DICTIONARY_HEADER_SIZE);
	auto symbol_table_dest = AlignValue<idx_t>(dictionary_dest + header.dict_size);
	auto string_lengths_dest = AlignValue<idx_t>(symbol_table_dest + header.symbol_table_size);
	vector<uint32_t> lengths(BitpackingPrimitives::RoundUpToAlgorithmGroupSize<idx_t>(header.dict_count));
	BitpackingPrimitives::UnPackBuffer<uint32_t>(data_ptr_cast(lengths.data()), base + string_lengths_dest,
	                                             header.dict_count, header.string_lengths_width);
	idx_t offset = 0;
	for (idx_t i = 0; i < local; i++) {
		offset += lengths[i];
	}
	auto length = lengths[local];
	auto str = char_ptr_cast(base + dictionary_dest + offset);
	if (header.mode == dict_fsst::DictFSSTMode::DICTIONARY) {
		return StringVector::AddStringOrBlob(result, str, length);
	}
	if (length == 0) {
		return string_t(nullptr, 0);
	}
	duckdb_fsst_decoder_t decoder;
	duckdb_fsst_import(&decoder, base + symbol_table_dest);
	return FSSTPrimitives::DecompressValue(&decoder, StringVector::GetStringBuffer(result), str, length);
}

//===--------------------------------------------------------------------===//
// The registry of stored columns
//===--------------------------------------------------------------------===//
namespace {
struct StoredTable {
	vector<PersistedColumn> columns;
	//! an alter made the entries unreadable until the next checkpoint
	bool invalidated = false;
	//! the read translations, by storage index (created on first use)
	unordered_map<idx_t, shared_ptr<PersistedTranslations>> read;
	BlockManager *block_manager = nullptr;
	DatabaseInstance *db = nullptr;
};
struct PersistedRegistry {
	mutex lock;
	unordered_map<const DataTableInfo *, StoredTable> tables;
	unordered_map<const PersistentTableData *, vector<PersistedColumn>> stash;
};
PersistedRegistry &Persisted() {
	// never destroyed, as the dictionary registry
	static auto registry = new PersistedRegistry();
	return *registry;
}
} // namespace

void StashPersisted(const PersistentTableData &data, vector<PersistedColumn> columns) {
	auto &registry = Persisted();
	lock_guard<mutex> guard(registry.lock);
	registry.stash[&data] = std::move(columns);
}

void AdoptPersisted(const PersistentTableData &data, const DataTableInfo &info) {
	auto &registry = Persisted();
	lock_guard<mutex> guard(registry.lock);
	auto found = registry.stash.find(&data);
	if (found == registry.stash.end()) {
		return;
	}
	auto &table = registry.tables[&info];
	table.columns = std::move(found->second);
	table.read.clear();
	table.invalidated = false;
	registry.stash.erase(found);
}

void InvalidatePersisted(const DataTableInfo &info) noexcept {
	try {
		auto &registry = Persisted();
		lock_guard<mutex> guard(registry.lock);
		auto found = registry.tables.find(&info);
		if (found != registry.tables.end()) {
			found->second.invalidated = true;
			found->second.read.clear();
		}
	} catch (std::exception &) { // NOLINT: called from constructors that must not fail here
	}
}

void ReleasePersisted(const DataTableInfo &info) noexcept {
	try {
		auto &registry = Persisted();
		lock_guard<mutex> guard(registry.lock);
		registry.tables.erase(&info);
	} catch (std::exception &) { // NOLINT: a destructor's caller
	}
}

void CommitDropPersisted(const DataTableInfo &info, BlockManager &block_manager) {
	vector<PersistedColumn> dropped;
	{
		auto &registry = Persisted();
		lock_guard<mutex> guard(registry.lock);
		auto found = registry.tables.find(&info);
		if (found == registry.tables.end()) {
			return;
		}
		dropped = std::move(found->second.columns);
		registry.tables.erase(found);
	}
	for (auto &column : dropped) {
		for (auto block_id : column.blocks) {
			block_manager.MarkBlockAsModified(block_id);
		}
	}
}

shared_ptr<PersistedTranslations> FindPersistedTranslations(const DataTableInfo &info, idx_t storage_index) {
	if (!PersistedTranslationsEnabled()) {
		return nullptr;
	}
	auto &registry = Persisted();
	lock_guard<mutex> guard(registry.lock);
	auto found = registry.tables.find(&info);
	if (found == registry.tables.end()) {
		return nullptr;
	}
	auto &table = found->second;
	if (table.invalidated) {
		return nullptr;
	}
	auto read = table.read.find(storage_index);
	if (read != table.read.end()) {
		return read->second;
	}
	for (auto &column : table.columns) {
		if (column.storage_index == storage_index) {
			auto &block_manager = const_cast<DataTableInfo &>(info).GetIOManager().GetBlockManagerForRowData();
			auto result = make_shared_ptr<PersistedTranslations>(block_manager.buffer_manager.GetDatabase(),
			                                                     block_manager, column);
			table.read[storage_index] = result;
			return result;
		}
	}
	return nullptr;
}

//===--------------------------------------------------------------------===//
// The build at checkpoint
//===--------------------------------------------------------------------===//
namespace {
static constexpr idx_t DEDUP_PARTITION_BITS = 10;
static constexpr idx_t DEDUP_PARTITIONS = 1ULL << DEDUP_PARTITION_BITS;
static constexpr idx_t SEGMENTS_PER_TASK = 8;

//! The build's working set - the partitions' strings, hashes, first occurrences, codes and tables, the segments' packed
//! entries, bitmaps and old codes, then the blob - is reserved in the buffer pool as it grows, in whole steps: the pool
//! evicts to make room for a step, and a step that would take the build above its ceiling, or that the pool cannot make
//! room for, refuses the column, which is then read through its segments' dictionaries as a column without stored
//! translations. The load itself never fails on it
static constexpr idx_t BUILD_RESERVATION_STEP = 64ULL * 1024ULL * 1024ULL;

//! A build's ceiling: half of the memory_limit, and below SMALL_MEMORY_LIMIT a quarter of it (the global dictionary's
//! budget there: the rest of a small pool stays with the checkpoint and the statements beside it)
static idx_t BuildCeiling(idx_t memory_limit) {
	return memory_limit < SMALL_MEMORY_LIMIT ? memory_limit / 4 : memory_limit / 2;
}

struct BuildReservation {
	BuildReservation(BufferManager &buffer_manager_p, bool &pressure_p)
	    : buffer_manager(buffer_manager_p), pressure(pressure_p) {
	}
	~BuildReservation() {
		if (reserved) {
			buffer_manager.FreeReservedMemory(reserved);
		}
	}
	//! `added` more bytes of working set: the steps that cover them reserved first (OutOfMemoryException: refused)
	void Account(idx_t added) {
		lock_guard<mutex> guard(lock);
		footprint += added;
		while (reserved < footprint) {
			const idx_t limit = buffer_manager.GetMaxMemory();
			const idx_t after = reserved + BUILD_RESERVATION_STEP;
			if (after > BuildCeiling(limit)) {
				throw OutOfMemoryException("Stored column translations: the build refused under memory pressure (%llu of "
				                           "memory_limit %llu bytes)",
				                           (unsigned long long)after, (unsigned long long)limit);
			}
			const bool evicts = buffer_manager.GetUsedMemory() + BUILD_RESERVATION_STEP > limit;
			buffer_manager.ReserveMemory(BUILD_RESERVATION_STEP);
			reserved = after;
			if (evicts) {
				// the blocks evicted for the step went back to the allocator: return them to the system before the build
				// allocates (the build's vectors come from the system heap, not the allocator's arenas)
				Allocator::FlushAll();
				pressure = true;
			}
		}
	}
	//! The working set is now `target` bytes (the transient state freed): the whole steps above it are released
	void ShrinkTo(idx_t target) {
		lock_guard<mutex> guard(lock);
		footprint = target;
		const idx_t keep = (target + BUILD_RESERVATION_STEP - 1) / BUILD_RESERVATION_STEP * BUILD_RESERVATION_STEP;
		if (keep < reserved) {
			buffer_manager.FreeReservedMemory(reserved - keep);
			reserved = keep;
		}
	}

	BufferManager &buffer_manager;
	//! set when a step had the pool evict
	bool &pressure;
	mutex lock;
	idx_t footprint = 0;
	idx_t reserved = 0;
};

unique_ptr<TaskExecutor> MakeExecutor(optional_ptr<ClientContext> context, DatabaseInstance &db) {
	if (context) {
		return make_uniq<TaskExecutor>(*context);
	}
	return make_uniq<TaskExecutor>(TaskScheduler::GetScheduler(db));
}

//! One hash partition of the column's distinct strings, filled concurrently under its own lock; each string keeps its
//! first occurrence (segment, local entry) and, once numbered, its code
struct DedupPartition {
	mutex lock;
	unsafe_unique_array<uint32_t> table;
	idx_t capacity = 0;
	vector<string_t> strings;
	vector<hash_t> hashes;
	vector<uint64_t> first;
	vector<uint32_t> codes;
	buffer_ptr<VectorStringBuffer> arena;

	void Resize(idx_t new_capacity) {
		auto new_table = make_unsafe_uniq_array<uint32_t>(new_capacity);
		memset(new_table.get(), 0, new_capacity * sizeof(uint32_t));
		const idx_t mask = new_capacity - 1;
		for (idx_t id = 0; id < strings.size(); id++) {
			idx_t slot = hashes[id] & mask;
			while (new_table[slot] != 0) {
				slot = (slot + 1) & mask;
			}
			new_table[slot] = UnsafeNumericCast<uint32_t>(id + 1);
		}
		table = std::move(new_table);
		capacity = new_capacity;
	}

	//! The string's id within the partition (added when new); its first occurrence lowered to `occurrence`; `added`
	//! grows by the memory a new string takes
	uint32_t Insert(const string_t &value, hash_t hash, uint64_t occurrence, idx_t &added) {
		lock_guard<mutex> guard(lock);
		if (!table) {
			arena = make_buffer<VectorStringBuffer>();
			Resize(256);
			added += 256 * sizeof(uint32_t);
		}
		const idx_t mask = capacity - 1;
		idx_t slot = hash & mask;
		while (true) {
			const auto stored = table[slot];
			if (stored == 0) {
				break;
			}
			const auto id = stored - 1;
			if (hashes[id] == hash && strings[id] == value) {
				first[id] = MinValue<uint64_t>(first[id], occurrence);
				return id;
			}
			slot = (slot + 1) & mask;
		}
		const auto id = UnsafeNumericCast<uint32_t>(strings.size());
		const idx_t strings_capacity = strings.capacity();
		const idx_t hashes_capacity = hashes.capacity();
		const idx_t first_capacity = first.capacity();
		strings.push_back(value.IsInlined() ? value : arena->AddString(value));
		hashes.push_back(hash);
		first.push_back(occurrence);
		table[slot] = id + 1;
		// the memory this insert took: the string's arena bytes, and the vectors' growth when they reallocated
		added += value.IsInlined() ? 0 : value.GetSize();
		added += (strings.capacity() - strings_capacity) * sizeof(string_t) +
		         (hashes.capacity() - hashes_capacity) * sizeof(hash_t) +
		         (first.capacity() - first_capacity) * sizeof(uint64_t);
		if (strings.size() * 2 > capacity) {
			added += capacity * 2 * sizeof(uint32_t);
			Resize(capacity * 2);
		}
		return id;
	}
};

//! One segment's local entries as (partition << 32 | id) until the partitions are numbered
struct DedupSegment {
	uint32_t dict_count = 0;
	idx_t rows = 0;
	unsafe_unique_array<uint64_t> packed;
	unsafe_unique_array<uint64_t> bitmap;
	uint32_t new_count = 0;
	uint32_t start_code = 0;
	vector<uint32_t> old_codes;
	uint8_t old_width = 0;
};

struct CheckpointBuildState {
	CheckpointBuildState(BufferManager &buffer_manager_p, bool &pressure)
	    : buffer_manager(buffer_manager_p), reservation(buffer_manager_p, pressure) {
	}
	BufferManager &buffer_manager;
	BuildReservation reservation;
	//! a segment without a code array (FSST_ONLY): the column is not stored as translations
	atomic<bool> refused {false};
	//! a reservation step refused: the column is not stored as translations
	atomic<bool> refused_memory {false};

	//! `added` bytes reserved from a task: false (and every task stops) when refused
	bool Account(idx_t added) {
		try {
			reservation.Account(added);
			return true;
		} catch (OutOfMemoryException &) {
			refused_memory = true;
			return false;
		}
	}
	vector<reference<ColumnSegment>> segments;
	vector<DedupSegment> work;
	unsafe_unique_array<DedupPartition> partitions;
};

uint64_t Occurrence(idx_t segment, idx_t local) {
	return (static_cast<uint64_t>(segment) << 32) | local;
}

class DedupTask : public BaseExecutorTask {
public:
	DedupTask(TaskExecutor &executor, CheckpointBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t k = begin; k < end; k++) {
			if (state.refused.load(std::memory_order_relaxed) || state.refused_memory.load(std::memory_order_relaxed)) {
				return;
			}
			auto &segment = state.segments[k].get();
			auto &work = state.work[k];
			dict_fsst::CompressedStringScanState scan_state(segment, state.buffer_manager.Pin(segment.block));
			scan_state.Initialize(false);
			if (scan_state.mode == dict_fsst::DictFSSTMode::FSST_ONLY) {
				state.refused = true;
				return;
			}
			const idx_t dict_count = scan_state.dict_count;
			if (!state.Account(dict_count * sizeof(uint64_t))) {
				return;
			}
			work.dict_count = UnsafeNumericCast<uint32_t>(dict_count);
			work.rows = segment.count.load();
			work.packed = make_unsafe_uniq_array_uninitialized<uint64_t>(dict_count);
			work.packed[0] = 0;
			Vector decoded(LogicalType::VARCHAR, 1);
			uint32_t offset = 0;
			idx_t added = 0;
			for (idx_t local = 0; local < dict_count; local++) {
				const auto length = scan_state.string_lengths[local];
				if (local > 0) {
					const auto value = scan_state.FetchEntry(decoded, offset, local);
					const auto hash = Hash(value);
					const auto partition = hash >> (64 - DEDUP_PARTITION_BITS);
					const auto id = state.partitions[partition].Insert(value, hash, Occurrence(k, local), added);
					work.packed[local] = (static_cast<uint64_t>(partition) << 32) | id;
				}
				offset += length;
			}
			if (!state.Account(added)) {
				return;
			}
		}
	}
	string TaskType() const override {
		return "StoredTranslationDedupTask";
	}

private:
	CheckpointBuildState &state;
	idx_t begin;
	idx_t end;
};

//! Per segment: the bitmap of the local entries that occur first there, and their number
class FirstTask : public BaseExecutorTask {
public:
	FirstTask(TaskExecutor &executor, CheckpointBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t k = begin; k < end; k++) {
			auto &work = state.work[k];
			const idx_t words = (work.dict_count + 63) / 64;
			work.bitmap = make_unsafe_uniq_array<uint64_t>(MaxValue<idx_t>(words, 1));
			memset(work.bitmap.get(), 0, MaxValue<idx_t>(words, 1) * sizeof(uint64_t));
			uint32_t count = 0;
			for (idx_t local = 1; local < work.dict_count; local++) {
				const auto packed = work.packed[local];
				auto &partition = state.partitions[packed >> 32];
				if (partition.first[packed & 0xFFFFFFFFULL] == Occurrence(k, local)) {
					work.bitmap[local / 64] |= 1ULL << (local % 64);
					count++;
				}
			}
			work.new_count = count;
		}
	}
	string TaskType() const override {
		return "StoredTranslationFirstTask";
	}

private:
	CheckpointBuildState &state;
	idx_t begin;
	idx_t end;
};

//! Per segment: the codes of its new entries (start code + rank), written into the partitions
class NumberTask : public BaseExecutorTask {
public:
	NumberTask(TaskExecutor &executor, CheckpointBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t k = begin; k < end; k++) {
			auto &work = state.work[k];
			uint32_t next = work.start_code;
			for (idx_t local = 1; local < work.dict_count; local++) {
				if ((work.bitmap[local / 64] >> (local % 64)) & 1) {
					const auto packed = work.packed[local];
					state.partitions[packed >> 32].codes[packed & 0xFFFFFFFFULL] = next++;
				}
			}
		}
	}
	string TaskType() const override {
		return "StoredTranslationNumberTask";
	}

private:
	CheckpointBuildState &state;
	idx_t begin;
	idx_t end;
};

//! Per segment: its old entries' codes (all numbered by now)
class OldTask : public BaseExecutorTask {
public:
	OldTask(TaskExecutor &executor, CheckpointBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t k = begin; k < end; k++) {
			auto &work = state.work[k];
			uint32_t max_old = 0;
			work.old_codes.reserve(
			    BitpackingPrimitives::RoundUpToAlgorithmGroupSize<idx_t>(work.dict_count - 1 - work.new_count));
			for (idx_t local = 1; local < work.dict_count; local++) {
				if (!((work.bitmap[local / 64] >> (local % 64)) & 1)) {
					const auto packed = work.packed[local];
					const auto code = state.partitions[packed >> 32].codes[packed & 0xFFFFFFFFULL];
					work.old_codes.push_back(code);
					max_old = MaxValue(max_old, code);
				}
			}
			work.old_width = WidthOf(max_old);
			work.packed.reset();
		}
	}
	string TaskType() const override {
		return "StoredTranslationOldTask";
	}

private:
	CheckpointBuildState &state;
	idx_t begin;
	idx_t end;
};

template <class TASK>
void RunTasks(optional_ptr<ClientContext> context, DatabaseInstance &db, CheckpointBuildState &state) {
	auto executor = MakeExecutor(context, db);
	const idx_t count = state.segments.size();
	for (idx_t begin = 0; begin < count; begin += SEGMENTS_PER_TASK) {
		executor->ScheduleTask(make_uniq<TASK>(*executor, state, begin, MinValue(count, begin + SEGMENTS_PER_TASK)));
	}
	executor->WorkOnTasks();
}

//! The column's segments in collection order and their identity; false when the column cannot be stored as
//! translations: a segment that is not a persistent DICT_FSST segment (a split one with a code array), or a column with
//! updates. `entry_bound` bounds the column's distinct strings from above
bool CollectSegments(RowGroupCollection &row_groups, idx_t storage_index, vector<reference<ColumnSegment>> &segments,
                     uint64_t &identity, idx_t &rows, idx_t &entry_bound) {
	identity = Hash<uint64_t>(storage_index);
	rows = 0;
	entry_bound = 0;
	auto tree = row_groups.GetRowGroups();
	for (auto row_group_node = tree->GetRootSegment(); row_group_node;
	     row_group_node = tree->GetNextSegment(*row_group_node)) {
		auto &column = row_group_node->GetNode().GetRawColumnData(storage_index);
		if (column.HasUpdates()) {
			return false;
		}
		auto &segment_tree = column.GetSegmentTree();
		for (auto segment_node = segment_tree.GetRootSegment(); segment_node;
		     segment_node = segment_tree.GetNextSegment(*segment_node)) {
			auto &segment = segment_node->GetNode();
			if (segment.GetCompressionFunction().type != CompressionType::COMPRESSION_DICT_FSST ||
			    segment.segment_type != ColumnSegmentType::PERSISTENT || segment.GetBlockId() == INVALID_BLOCK) {
				return false;
			}
			auto split = dict_fsst::SegmentSplit(segment);
			if (split && split->mode == static_cast<uint8_t>(dict_fsst::DictFSSTMode::FSST_ONLY)) {
				return false;
			}
			identity = CombineHash(identity, Hash<int64_t>(segment.GetBlockId()));
			identity = CombineHash(identity, Hash<uint64_t>(segment.GetBlockOffset()));
			identity = CombineHash(identity, Hash<uint64_t>(segment.count.load()));
			identity = CombineHash(identity, Hash<uint64_t>(split ? split->dict_count : 0));
			rows += segment.count.load();
			entry_bound += split ? split->dict_count - 1 : segment.count.load();
			segments.push_back(segment);
		}
	}
	return !segments.empty();
}

//! Build one column's translations over `segments` and write its blob; false when refused
bool BuildAtCheckpoint(optional_ptr<ClientContext> context, DatabaseInstance &db, BlockManager &block_manager,
                       idx_t storage_index, uint64_t identity, idx_t rows, vector<reference<ColumnSegment>> segments,
                       PersistedColumn &result, bool &pressure) {
	auto &buffer_manager = BufferManager::GetBufferManager(db);
	CheckpointBuildState state(buffer_manager, pressure);
	state.segments = std::move(segments);
	state.reservation.Account(state.segments.size() * (sizeof(DedupSegment) + sizeof(reference<ColumnSegment>)) +
	                          DEDUP_PARTITIONS * sizeof(DedupPartition));
	state.work.resize(state.segments.size());
	state.partitions = make_unsafe_uniq_array<DedupPartition>(DEDUP_PARTITIONS);
	RunTasks<DedupTask>(context, db, state);
	if (state.refused) {
		return false;
	}
	if (state.refused_memory) {
		throw OutOfMemoryException("Stored column translations: the build refused under memory pressure (%llu bytes "
		                           "of working set)",
		                           (unsigned long long)state.reservation.footprint);
	}
	idx_t distinct = 0;
	for (idx_t p = 0; p < DEDUP_PARTITIONS; p++) {
		distinct += state.partitions[p].strings.size();
	}
	// the admission rule's distinct floors: at least kPersistedTranslationMinDistinct, at most rows / k
	if (distinct < idx_t(kPersistedTranslationMinDistinct) ||
	    double(distinct) > double(rows) * kPersistedTranslationMaxDistinctShare ||
	    distinct + 1 > idx_t(NumericLimits<int32_t>::Maximum())) {
		return false;
	}
	// the first occurrences: per segment its new entries, then the start codes, then every code (the bitmaps and the
	// codes reserved before they are allocated)
	idx_t bitmap_bytes_total = 0;
	for (auto &work : state.work) {
		bitmap_bytes_total += MaxValue<idx_t>((work.dict_count + 63) / 64, 1) * sizeof(uint64_t);
	}
	state.reservation.Account(bitmap_bytes_total + distinct * sizeof(uint32_t));
	RunTasks<FirstTask>(context, db, state);
	uint32_t next = 1;
	for (auto &work : state.work) {
		work.start_code = next;
		next += work.new_count;
	}
	if (next != distinct + 1) {
		throw InternalException("Stored column translations: %llu first occurrences for %llu distinct strings",
		                        (unsigned long long)(next - 1), (unsigned long long)distinct);
	}
	for (idx_t p = 0; p < DEDUP_PARTITIONS; p++) {
		state.partitions[p].codes.resize(state.partitions[p].strings.size());
	}
	RunTasks<NumberTask>(context, db, state);
	idx_t old_bytes_total = 0;
	for (auto &work : state.work) {
		old_bytes_total += BitpackingPrimitives::RoundUpToAlgorithmGroupSize<idx_t>(work.dict_count - 1 - work.new_count) *
		                   sizeof(uint32_t);
	}
	state.reservation.Account(old_bytes_total);
	RunTasks<OldTask>(context, db, state);
	// the code of the empty string, if the column holds it
	uint32_t empty_code = 0;
	{
		string_t empty("", 0);
		const auto hash = Hash(empty);
		auto &partition = state.partitions[hash >> (64 - DEDUP_PARTITION_BITS)];
		for (idx_t id = 0; id < partition.strings.size(); id++) {
			if (partition.hashes[id] == hash && partition.strings[id] == empty) {
				empty_code = partition.codes[id];
			}
		}
	}
	state.partitions.reset();
	// what is left of the working set: the segments' records, bitmaps and old codes (their packed entries are gone)
	state.reservation.ShrinkTo(state.segments.size() * (sizeof(DedupSegment) + sizeof(reference<ColumnSegment>)) +
	                           bitmap_bytes_total + old_bytes_total);

	// the blob, laid out so that no record crosses a block boundary
	const idx_t block_size = block_manager.GetBlockSize();
	vector<idx_t> entry_positions(state.work.size());
	vector<idx_t> data_positions(state.work.size());
	idx_t position = sizeof(BlobHeader);
	for (idx_t k = 0; k < state.work.size(); k++) {
		position = Place(position, ENTRY_SIZE, block_size);
		entry_positions[k] = position;
		position += ENTRY_SIZE;
	}
	for (idx_t k = 0; k < state.work.size(); k++) {
		auto &work = state.work[k];
		const idx_t bytes = BitmapBytes(work.dict_count) + OldBytes(work.old_codes.size(), work.old_width);
		if (bytes > block_size) {
			return false;
		}
		position = Place(position, bytes, block_size);
		data_positions[k] = position;
		position += bytes;
	}
	const idx_t blob_bytes = position;
	const idx_t block_count = (blob_bytes + block_size - 1) / block_size;
	state.reservation.Account(block_count * block_size);
	auto blob = make_unsafe_uniq_array<data_t>(block_count * block_size);
	memset(blob.get(), 0, block_count * block_size);
	BlobHeader header;
	memset(&header, 0, sizeof(header));
	header.magic = BLOB_MAGIC;
	header.version = BLOB_VERSION;
	header.empty_code = empty_code;
	header.storage_index = storage_index;
	header.count = distinct + 1;
	header.segments = state.work.size();
	header.rows = rows;
	memcpy(blob.get(), &header, sizeof(header));
	for (idx_t k = 0; k < state.work.size(); k++) {
		auto &segment = state.segments[k].get();
		auto &work = state.work[k];
		BlobEntry entry;
		memset(&entry, 0, sizeof(entry));
		entry.block_id = segment.GetBlockId();
		entry.offset = NumericCast<uint32_t>(segment.GetBlockOffset());
		entry.dict_count = work.dict_count;
		entry.start_code = work.start_code;
		entry.new_count = work.new_count;
		entry.data_offset = data_positions[k];
		entry.rows = work.rows;
		entry.old_width = work.old_width;
		memcpy(blob.get() + entry_positions[k], &entry, sizeof(entry));
		const idx_t bitmap_bytes = BitmapBytes(work.dict_count);
		memcpy(blob.get() + data_positions[k], work.bitmap.get(), bitmap_bytes);
		if (!work.old_codes.empty() && work.old_width > 0) {
			work.old_codes.resize(BitpackingPrimitives::RoundUpToAlgorithmGroupSize<idx_t>(work.old_codes.size()), 0);
			BitpackingPrimitives::PackBuffer<uint32_t, false>(blob.get() + data_positions[k] + bitmap_bytes,
			                                                  work.old_codes.data(), work.old_codes.size(),
			                                                  work.old_width);
		}
	}
	result.storage_index = storage_index;
	result.identity = identity;
	result.count = distinct + 1;
	result.segments = state.work.size();
	result.rows = rows;
	result.blob_bytes = blob_bytes;
	for (idx_t b = 0; b < block_count; b++) {
		const auto block_id = block_manager.GetFreeBlockIdForCheckpoint();
		auto block = block_manager.CreateBlock(block_id, nullptr);
		memcpy(block->buffer, blob.get() + b * block_size, block_size);
		block_manager.Write(QueryContext(context), *block, block_id);
		result.blocks.push_back(block_id);
	}
	return true;
}
} // namespace

vector<PersistedColumn> PersistAtCheckpoint(optional_ptr<ClientContext> context, DatabaseInstance &db,
                                            DataTableInfo &info, RowGroupCollection &collection,
                                            const vector<ColumnDefinition> &columns, BlockManager &block_manager,
                                            bool unchanged) {
	vector<PersistedColumn> existing;
	{
		auto &registry = Persisted();
		lock_guard<mutex> guard(registry.lock);
		auto found = registry.tables.find(&info);
		if (found != registry.tables.end()) {
			existing = found->second.columns;
		}
	}
	if (unchanged) {
		return existing;
	}
	const bool build = PersistedTranslationsEnabled() && dict_fsst::SplitSegmentsEnabled(block_manager);
	vector<PersistedColumn> result;
	vector<bool> carried(existing.size(), false);
	for (auto &column : columns) {
		if (column.Type().id() != LogicalTypeId::VARCHAR || !StringType::GetCollation(column.Type()).empty()) {
			continue;
		}
		const auto storage_index = column.StorageOid();
		vector<reference<ColumnSegment>> segments;
		uint64_t identity;
		idx_t rows;
		idx_t entry_bound;
		if (!CollectSegments(collection, storage_index, segments, identity, rows, entry_bound)) {
			continue;
		}
		bool kept = false;
		for (idx_t e = 0; e < existing.size(); e++) {
			if (!carried[e] && existing[e].storage_index == storage_index && existing[e].identity == identity &&
			    existing[e].segments == segments.size()) {
				carried[e] = true;
				result.push_back(existing[e]);
				kept = true;
				break;
			}
		}
		// the admission rule's row floor, and the distinct floor on its upper bound (the local entries' sum) before any
		// dictionary is read
		if (kept || !build || rows < idx_t(kPersistedTranslationMinRows) ||
		    entry_bound < idx_t(kPersistedTranslationMinDistinct)) {
			continue;
		}
		PersistedColumn persisted;
		bool built = false;
		bool pressure = false;
		try {
			built = BuildAtCheckpoint(context, db, block_manager, storage_index, identity, rows, std::move(segments),
			                          persisted, pressure);
		} catch (OutOfMemoryException &) {
			// the stored translations are optional: a column whose build does not fit is read as before
			built = false;
		} catch (std::bad_alloc &) {
			built = false;
		}
		if (pressure) {
			// a build that had the pool evict: its freed working set goes back to the system before the next column
			Allocator::FlushAll();
		}
		if (built) {
			result.push_back(std::move(persisted));
		}
	}
	// an entry not carried belongs to segments this checkpoint no longer holds: its blocks are free after it
	for (idx_t e = 0; e < existing.size(); e++) {
		if (carried[e]) {
			continue;
		}
		for (auto block_id : existing[e].blocks) {
			block_manager.MarkBlockAsModified(block_id);
		}
	}
	{
		auto &registry = Persisted();
		lock_guard<mutex> guard(registry.lock);
		auto &table = registry.tables[&info];
		table.columns = result;
		table.read.clear();
		table.invalidated = false;
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Codes-only reads
//===--------------------------------------------------------------------===//
//! A codes-only read of a segment its column's lazily linked translations do not hold (LinkLoadedColumn)
[[noreturn]] static void ThrowUnlinkedSegment(const ScanPublication::Column &column) {
	throw InternalException("Stored column translations: a segment of \"%s\" is not in the translations its plan reads",
	                        column.codes_only_dict ? column.codes_only_dict->column_name : column.name);
}

bool SegmentReadsCodesOnly(ColumnSegment &segment) {
	auto publication = ThreadPublication();
	if (!publication) {
		return false;
	}
	idx_t entry;
	auto translations = dict_fsst::SegmentTranslationLink(segment, entry);
	if (!translations) {
		return false;
	}
	auto column = publication->Find(translations->storage_index);
	const bool codes_only = column && column->ReadsCodesOnly() && column->codes_only.get() == translations.get();
	if (codes_only && entry == DConstants::INVALID_INDEX) {
		ThrowUnlinkedSegment(*column);
	}
	return codes_only;
}

shared_ptr<SegmentTranslation> CodesOnlyTranslation(ColumnSegment &segment) {
	auto publication = ThreadPublication();
	if (!publication) {
		return nullptr;
	}
	idx_t entry_index;
	auto translations = dict_fsst::SegmentTranslationLink(segment, entry_index);
	if (!translations) {
		return nullptr;
	}
	auto column = publication->Find(translations->storage_index);
	if (!column || !column->ReadsCodesOnly()) {
		return nullptr;
	}
	if (column->codes_only.get() != translations.get() || !column->codes_only_dict) {
		// planned codes-only over other translations than this segment's (a checkpoint since planning): never read a
		// segment past its plan
		ThrowStaleTranslations(column->codes_only_dict ? column->codes_only_dict->column_name : column->name);
	}
	if (entry_index == DConstants::INVALID_INDEX) {
		ThrowUnlinkedSegment(*column);
	}
	auto &entry = translations->Entries()[entry_index];
	auto codes = make_unsafe_uniq_array_uninitialized<uint32_t>(MaxValue<idx_t>(entry.dict_count, 1));
	translations->Decode(entry_index, codes.get());
	auto result = make_shared_ptr<SegmentTranslation>(column->codes_only_dict, std::move(codes), entry.dict_count);
	result->filter_only = column->filter_only;
	return result;
}

static bool IsEmptyStringConstant(const ConstantFilter &filter) {
	return filter.constant.type().id() == LogicalTypeId::VARCHAR && !filter.constant.IsNull() &&
	       StringValue::Get(filter.constant).empty();
}

bool CodeTranslatable(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant = filter.Cast<ConstantFilter>();
		return (constant.comparison_type == ExpressionType::COMPARE_EQUAL ||
		        constant.comparison_type == ExpressionType::COMPARE_NOTEQUAL) &&
		       IsEmptyStringConstant(constant);
	}
	case TableFilterType::IS_NULL:
	case TableFilterType::IS_NOT_NULL:
	case TableFilterType::OPTIONAL_FILTER:
		return true;
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction = filter.Cast<ConjunctionAndFilter>();
		for (auto &child : conjunction.child_filters) {
			if (!CodeTranslatable(*child)) {
				return false;
			}
		}
		return true;
	}
	default:
		return false;
	}
}

//! Whether a non-NULL or NULL code passes (CodeTranslatable shapes; an optional filter passes everything)
static bool CodePasses(const TableFilter &filter, uint32_t code, uint32_t empty_code) {
	switch (filter.filter_type) {
	case TableFilterType::CONSTANT_COMPARISON: {
		auto &constant = filter.Cast<ConstantFilter>();
		if (code == 0) {
			return false;
		}
		const bool is_empty = empty_code != 0 && code == empty_code;
		return constant.comparison_type == ExpressionType::COMPARE_EQUAL ? is_empty : !is_empty;
	}
	case TableFilterType::IS_NULL:
		return code == 0;
	case TableFilterType::IS_NOT_NULL:
		return code != 0;
	case TableFilterType::OPTIONAL_FILTER:
		return true;
	case TableFilterType::CONJUNCTION_AND: {
		auto &conjunction = filter.Cast<ConjunctionAndFilter>();
		for (auto &child : conjunction.child_filters) {
			if (!CodePasses(*child, code, empty_code)) {
				return false;
			}
		}
		return true;
	}
	default:
		throw InternalException("CodePasses: a filter that is not decided on codes");
	}
}

//! Every codes-only filter path decides a code by its class through CodePasses, which reads any constant comparison as
//! the empty-string test: a filter CodeTranslatable does not admit is refused before, never mis-selected
static void CheckCodeTranslatable(const TableFilter &filter) {
	if (!CodeTranslatable(filter)) {
		throw InternalException("dict_global: a filter that is not decided on codes reached a codes-only column");
	}
}

void CodeClassVerdicts(const TableFilter &filter, uint32_t empty_code, bool &null_passes, bool &empty_passes,
                       bool &other_passes) {
	CheckCodeTranslatable(filter);
	// a code stands for its class: 0 (NULL), the empty string's, any other (a non-zero code that is not the empty one)
	null_passes = CodePasses(filter, 0, empty_code);
	empty_passes = empty_code != 0 && CodePasses(filter, empty_code, empty_code);
	other_passes = CodePasses(filter, empty_code == 1 ? 2 : 1, empty_code);
}

void FilterCodes(PersistedTranslations &translations, const TableFilter &filter, const sel_t *codes, idx_t count,
                 SelectionVector &sel, idx_t &sel_count) {
	CheckCodeTranslatable(filter);
	// every shape CodeTranslatable admits decides a code by its class alone - NULL (0), the empty string, any other
	const auto empty_code = translations.EmptyCode();
	const bool null_passes = CodePasses(filter, 0, empty_code);
	const bool empty_passes = empty_code != 0 && CodePasses(filter, empty_code, empty_code);
	// a code that is neither: the largest code if it is not the empty string's, else the one below it
	const uint32_t other = translations.Count() - 1 != empty_code ? uint32_t(translations.Count() - 1)
	                                                                : uint32_t(translations.Count() - 2);
	const bool other_passes = other != 0 && other != empty_code && CodePasses(filter, other, empty_code);
	SelectionVector new_sel(MaxValue<idx_t>(count, 1));
	idx_t approved = 0;
	if (!sel.IsSet() && sel_count == count && !null_passes && !empty_passes && other_passes) {
		// the common shape (<> '' or IS NOT NULL on a column holding ''): a row passes iff its code is neither
		for (idx_t row = 0; row < count; row++) {
			const auto code = codes[row];
			new_sel.set_index(approved, row);
			approved += code != 0 && code != empty_code;
		}
	} else {
		for (idx_t i = 0; i < sel_count; i++) {
			const auto row = sel.get_index(i);
			const auto code = codes[row];
			const bool pass = code == 0 ? null_passes : (code == empty_code ? empty_passes : other_passes);
			new_sel.set_index(approved, row);
			approved += pass;
		}
	}
	if (approved < count) {
		sel.Initialize(new_sel);
	}
	sel_count = approved;
}

bool FilterCodesOnlyVector(Vector &result, idx_t count, SelectionVector &sel, idx_t &sel_count,
                           const TableFilter &filter) {
	if (result.GetVectorType() != VectorType::DICTIONARY_VECTOR ||
	    result.GetType().InternalType() != PhysicalType::VARCHAR) {
		return false;
	}
	auto translations = CodesOnlyTranslationsOf(DictionaryVector::DictionaryId(result));
	if (!translations) {
		return false;
	}
	if (!CodeTranslatable(filter)) {
		throw InternalException("A filter that is not decided on codes over a codes-only vector");
	}
	auto &codes = DictionaryVector::SelVector(result);
	vector<sel_t> flat(MaxValue<idx_t>(count, 1));
	for (idx_t i = 0; i < count; i++) {
		flat[i] = UnsafeNumericCast<sel_t>(codes.get_index(i));
	}
	FilterCodes(*translations, filter, flat.data(), count, sel, sel_count);
	return true;
}

} // namespace dict_global
} // namespace duckdb
