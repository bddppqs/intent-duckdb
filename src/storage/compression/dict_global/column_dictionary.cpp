#include "duckdb/storage/compression/dict_global/column_dictionary.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/operator/aggregate/fused_integer_aggregate.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/compression/dict_fsst/decompression.hpp"
#include "duckdb/storage/compression/dict_fsst/filter_verdict_cache.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/table/column_data.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/row_group.hpp"
#include "duckdb/storage/table/row_group_collection.hpp"
#include "duckdb/storage/table/row_group_segment_tree.hpp"
#include "duckdb/common/tuning_defaults.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace duckdb {
namespace dict_global {

//===--------------------------------------------------------------------===//
// Compile-time defaults (duckdb/common/tuning_defaults.hpp)
//===--------------------------------------------------------------------===//
bool DictGlobalEnabled() {
	return kGlobalDictionary;
}

bool CodeKeysEnabled() {
	return DictGlobalEnabled() && kGlobalDictionaryCodeKeys;
}

//! FindTranslation answers from the segment's cache state when the answer there was taken at the current translation
//! generation, and asks the ObjectCache (under its process-wide lock) only otherwise; off (or without the segment's
//! cache state), every lookup goes to the ObjectCache
static bool XlateSegCacheEnabled() {
	return kSegmentTranslationCache;
}

bool FusedDistinctEnabled() {
	return kGlobalDictionaryFusedDistinct;
}

//! PlanCodeKeys refuses the code keys of an aggregate whose packed group width with them typed passes
//! FusedIntegerAggregate::MAXIMUM_KEY_BYTES; off, every width is typed
static bool WideGuardEnabled() {
	return kGlobalDictionaryWideKeyGuard;
}

bool BorrowChildEnabled() {
	return kSegmentTranslationChildHandle;
}

//! The parallel build (kGlobalDictionaryParallelBuild): a column's build is spread over the scheduler's threads
//! (BuildColumnParallel) when it has more than one task's worth of segments; off, the calling thread builds it alone
//! (BuildColumn)
static bool ParallelBuildEnabled() {
	static const bool enabled = kGlobalDictionaryParallelBuild;
	return enabled;
}

//! The estimate gate (kGlobalDictionaryEstimateGate): a build is not started when its statistics estimate
//! (approx-unique x ESTIMATED_BYTES_PER_UNIQUE) cannot fit the global-dictionary budget beside every reservation,
//! published and in flight; off, every unbuilt marked column is attempted and refused only when a reservation step
//! crosses the budget
static bool EstimateGateEnabled() {
	static const bool enabled = kGlobalDictionaryEstimateGate;
	return enabled;
}

//! Set once the first dictionary publishes: before that no segment lookup is taken
static atomic<bool> any_published {false};
//! All live global-dictionary reservations - the published dictionaries' and the in-flight builds' - in one account, so
//! that concurrent builds see each other: the refusal's budget holds for their sum, not for each build alone
static atomic<idx_t> total_reserved {0};
//! PublicationVersion(): bumped (release) after every event that can change what Published() returns - a publish, a
//! storage epoch bump, a table's entry released
static atomic<idx_t> publication_version {0};
//! advanced after every change of a translation entry (Publish's puts, ReleaseTranslation's
//! delete), so an answer a segment remembered under an older value is asked again
static atomic<idx_t> translation_generation {0};

//===--------------------------------------------------------------------===//
// Column dictionary
//===--------------------------------------------------------------------===//
ColumnDictionary::ColumnDictionary(DatabaseInstance &db_p, const RowGroupCollection &collection_p,
                                   idx_t storage_index_p, idx_t epoch_p, string column_name_p)
    : db(db_p), collection(&collection_p), storage_index(storage_index_p), epoch(epoch_p),
      column_name(std::move(column_name_p)), state(static_cast<uint8_t>(ColumnState::UNBUILT)) {
}

ColumnDictionary::~ColumnDictionary() {
	if (reserved) {
		total_reserved -= reserved;
		try {
			BufferManager::GetBufferManager(db).FreeReservedMemory(reserved);
		} catch (std::exception &) { // NOLINT: a destructor never throws
		}
	}
}

uint32_t ColumnDictionary::Lookup(const string_t &value, hash_t hash) const {
	if (!index) {
		return INVALID_CODE;
	}
	auto strings = FlatVector::GetData<string_t>(child->data);
	auto hashes = FlatVector::GetData<hash_t>(child->cached_hashes);
	idx_t slot = hash & index_mask;
	while (true) {
		auto code = index[slot];
		if (code == 0) {
			return INVALID_CODE;
		}
		if (hashes[code] == hash && strings[code] == value) {
			return code;
		}
		slot = (slot + 1) & index_mask;
	}
}

buffer_ptr<VectorChildBuffer> ChildHandle(const buffer_ptr<VectorChildBuffer> &child) {
	if (!BorrowChildEnabled() || !child) {
		return nullptr;
	}
	// the column's child is shared by every scan thread: a vector over it takes and drops a reference on its one control
	// block per chunk. The handle's own block holds one reference to the child for as long as any copy of the handle
	// lives, so the child's lifetime is unchanged and the per-chunk counting moves to this block
	auto owner = make_shared_ptr<buffer_ptr<VectorChildBuffer>>(child);
	return buffer_ptr<VectorChildBuffer>(owner, owner->get());
}

string TranslationKey(const string &segment_key) {
	return "globaldict-xlat-" + segment_key;
}

shared_ptr<SegmentTranslation> FindTranslation(ColumnSegment &segment) {
	if (!any_published.load(std::memory_order_acquire)) {
		return nullptr;
	}
	if (segment.GetCompressionFunction().type != CompressionType::COMPRESSION_DICT_FSST) {
		return nullptr;
	}
	auto &key = segment.GetDictionaryCacheKey();
	if (key.empty()) {
		return nullptr;
	}
	shared_ptr<SegmentTranslation> entry;
	if (XlateSegCacheEnabled()) {
		// the answer this segment remembered at the current generation (an entry or its absence),
		// else the ObjectCache's, remembered; the generation is read before the ObjectCache is asked
		const idx_t generation = translation_generation.load(std::memory_order_acquire);
		shared_ptr<ObjectCacheEntry> remembered;
		if (dict_fsst::LookupSegmentTranslation(segment, generation, remembered)) {
			if (remembered) {
				// only ever an entry Get<SegmentTranslation> returned
				entry = shared_ptr_cast<ObjectCacheEntry, SegmentTranslation>(std::move(remembered));
			}
		} else {
			entry = segment.db.GetObjectCache().Get<SegmentTranslation>(TranslationKey(key));
			dict_fsst::RememberSegmentTranslation(segment, generation, entry);
		}
	} else {
		entry = segment.db.GetObjectCache().Get<SegmentTranslation>(TranslationKey(key));
	}
	if (!entry || entry->dict->GetState() != ColumnState::PUBLISHED) {
		return nullptr;
	}
	return entry;
}

void ReleaseTranslation(DatabaseInstance &db, const string &segment_key) {
	if (!any_published.load(std::memory_order_acquire) || segment_key.empty()) {
		return;
	}
	db.GetObjectCache().Delete(TranslationKey(segment_key));
	if (XlateSegCacheEnabled()) {
		translation_generation.fetch_add(1, std::memory_order_acq_rel);
	}
}

//===--------------------------------------------------------------------===//
// The scan's publication
//===--------------------------------------------------------------------===//
optional_ptr<const ScanPublication::Column> ScanPublication::Find(idx_t storage_index) const {
	for (auto &column : columns) {
		if (column.storage_index == storage_index) {
			return &column;
		}
	}
	return nullptr;
}

bool ScanPublication::Publishes(idx_t storage_index) const {
	auto column = Find(storage_index);
	return column && !column->gated;
}

const ScanPublication *&ThreadPublication() {
	thread_local const ScanPublication *publication = nullptr;
	return publication;
}

ThreadPublicationScope::ThreadPublicationScope(const ScanPublication *publication) : previous(ThreadPublication()) {
	ThreadPublication() = publication;
}

ThreadPublicationScope::~ThreadPublicationScope() {
	ThreadPublication() = previous;
}

namespace {
//! A seq_scan's bind data carrying its publication: the base fields copied exactly as TableScanBindData::Copy
//! copies them, so every reader of the base (Cast<TableScanBindData>()) reads the same values
struct PublishingTableScanBindData : public TableScanBindData {
	explicit PublishingTableScanBindData(const TableScanBindData &other) : TableScanBindData(other.table) {
		is_index_scan = other.is_index_scan;
		is_create_index = other.is_create_index;
		column_ids = other.column_ids;
		order_options = other.order_options ? make_uniq<RowGroupOrderOptions>(*other.order_options) : nullptr;
	}
	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<PublishingTableScanBindData>(static_cast<const TableScanBindData &>(*this));
		copy->publication = publication;
		return std::move(copy);
	}

	shared_ptr<ScanPublication> publication;
};
} // namespace

shared_ptr<ScanPublication> ScanPublicationOf(const FunctionData *bind_data) {
	auto publishing = dynamic_cast<const PublishingTableScanBindData *>(bind_data);
	return publishing ? publishing->publication : nullptr;
}

shared_ptr<SegmentTranslation> FindScanTranslation(ColumnSegment &segment) {
	auto publication = ThreadPublication();
	if (!publication) {
		return nullptr;
	}
	auto translation = FindTranslation(segment);
	if (!translation || !publication->Publishes(translation->dict->storage_index)) {
		return nullptr;
	}
	return translation;
}

static idx_t NextEpoch() {
	static atomic<idx_t> epochs {0};
	return ++epochs;
}

TableDictionaries::TableDictionaries(const shared_ptr<DataTableInfo> &info) : owner(info), epoch(NextEpoch()) {
}

namespace {
struct TableRegistry {
	mutex lock;
	unordered_map<const DataTableInfo *, shared_ptr<TableDictionaries>> tables;
};
TableRegistry &Registry() {
	// never destroyed: a static destructor at exit would release dictionaries after their database is gone
	static auto registry = new TableRegistry();
	return *registry;
}
} // namespace

shared_ptr<TableDictionaries> TableEntry(const shared_ptr<DataTableInfo> &info, bool create) {
	auto &registry = Registry();
	// released after the guard: a stale entry's dictionaries free their buffers outside the lock
	shared_ptr<TableDictionaries> stale;
	shared_ptr<DataTableInfo> owner;
	lock_guard<mutex> guard(registry.lock);
	auto found = registry.tables.find(info.get());
	if (found != registry.tables.end()) {
		owner = found->second->owner.lock();
		if (owner && owner.get() == info.get()) {
			return found->second;
		}
		// the entry of a table that is gone (~DataTableInfo erases it; this is the generation check behind that)
		stale = std::move(found->second);
		registry.tables.erase(found);
	}
	if (!create) {
		return nullptr;
	}
	auto entry = make_shared_ptr<TableDictionaries>(info);
	registry.tables[info.get()] = entry;
	return entry;
}

void ReleaseTable(const DataTableInfo &info) noexcept {
	shared_ptr<TableDictionaries> released;
	try {
		auto &registry = Registry();
		lock_guard<mutex> guard(registry.lock);
		auto found = registry.tables.find(&info);
		if (found == registry.tables.end()) {
			return;
		}
		released = std::move(found->second);
		registry.tables.erase(found);
	} catch (std::exception &) { // NOLINT: a destructor's caller
		return;
	}
	publication_version.fetch_add(1, std::memory_order_release);
	released.reset();
}

void BumpEpoch(const shared_ptr<DataTableInfo> &info) {
	auto entry = TableEntry(info, false);
	if (entry) {
		entry->epoch.store(NextEpoch());
		publication_version.fetch_add(1, std::memory_order_release);
	}
}

shared_ptr<ColumnDictionary> Published(DataTable &table, idx_t storage_index) {
	auto entry = TableEntry(table.GetDataTableInfo(), false);
	if (!entry) {
		return nullptr;
	}
	lock_guard<mutex> guard(entry->lock);
	auto found = entry->columns.find(storage_index);
	if (found == entry->columns.end()) {
		return nullptr;
	}
	auto &dict = found->second;
	if (dict->epoch != entry->epoch.load() || dict->collection != table.GetRowGroupCollection().get() ||
	    dict->GetState() != ColumnState::PUBLISHED) {
		return nullptr;
	}
	return dict;
}

idx_t PublicationVersion() {
	return publication_version.load(std::memory_order_acquire);
}

//===--------------------------------------------------------------------===//
// The build
//===--------------------------------------------------------------------===//
namespace {

static constexpr idx_t RESERVATION_STEP = 64ULL * 1024ULL * 1024ULL;

//! The global-dictionary budget: every global-dictionary reservation together - published dictionaries' and builds'
//! in flight - stays within 60 % of the buffer pool's memory_limit, and below SMALL_MEMORY_LIMIT within a quarter of
//! it, the share the fused aggregate's budget takes (GetMaxMemory() / 4). The reservations trade memory for speed and
//! are never evicted: below SMALL_MEMORY_LIMIT the rest of the pool must stay available to the statements running
//! beside them, while on a larger pool a quarter refuses a large column's build under concurrent statements
static idx_t Budget(idx_t memory_limit) {
	return memory_limit < SMALL_MEMORY_LIMIT ? memory_limit / 4 : memory_limit / 10 * 6;
}

struct Reservation {
	explicit Reservation(BufferManager &buffer_manager_p) : buffer_manager(buffer_manager_p) {
	}
	~Reservation() {
		if (reserved) {
			buffer_manager.FreeReservedMemory(reserved);
			total_reserved -= reserved;
		}
	}
	void Grow(idx_t needed) {
		while (reserved < needed) {
			// the pressure refusal, before the step is reserved: the step would take this build above half of the
			// buffer pool's memory_limit, or every global-dictionary reservation (published and in flight, this build's
			// included) above the budget. The step is claimed in the shared account first, so two builds racing for the
			// last steps cannot both pass
			const idx_t limit = buffer_manager.GetMaxMemory();
			const idx_t after = reserved + RESERVATION_STEP;
			const idx_t total = total_reserved.fetch_add(RESERVATION_STEP) + RESERVATION_STEP;
			if (after > limit / 2 || total > Budget(limit)) {
				total_reserved -= RESERVATION_STEP;
				throw OutOfMemoryException("Global dictionary: the column dictionary's build refused under memory pressure (%llu of "
				                           "memory_limit %llu bytes)",
				                           (unsigned long long)after, (unsigned long long)limit);
			}
			try {
				buffer_manager.ReserveMemory(RESERVATION_STEP);
			} catch (...) {
				total_reserved -= RESERVATION_STEP;
				throw;
			}
			reserved = after;
		}
	}
	//! Release the whole steps above `target` (a build whose transient state is gone keeps what it publishes)
	void ShrinkTo(idx_t target) {
		const idx_t keep = (target + RESERVATION_STEP - 1) / RESERVATION_STEP * RESERVATION_STEP;
		if (keep >= reserved) {
			return;
		}
		const idx_t release = reserved - keep;
		buffer_manager.FreeReservedMemory(release);
		total_reserved -= release;
		reserved = keep;
	}
	idx_t Release() {
		auto result = reserved;
		reserved = 0;
		return result;
	}
	BufferManager &buffer_manager;
	idx_t reserved = 0;
};

struct PendingTranslation {
	string segment_key;
	unsafe_unique_array<uint32_t> codes;
	idx_t count;
};

enum class BuildOutcome { PUBLISHED, REFUSED_MEMORY, REFUSED_EMPTY, INTERRUPTED };


struct Builder {
	explicit Builder(BufferManager &buffer_manager) : reservation(buffer_manager) {
		arena = make_buffer<VectorStringBuffer>();
		slots.emplace_back(string_t(nullptr, 0));
		hashes.push_back(0);
		ResizeIndex(1ULL << 16);
	}

	void ResizeIndex(idx_t capacity) {
		auto new_index = make_unsafe_uniq_array<uint32_t>(capacity);
		memset(new_index.get(), 0, capacity * sizeof(uint32_t));
		auto mask = capacity - 1;
		for (idx_t code = 1; code < slots.size(); code++) {
			idx_t slot = hashes[code] & mask;
			while (new_index[slot] != 0) {
				slot = (slot + 1) & mask;
			}
			new_index[slot] = UnsafeNumericCast<uint32_t>(code);
		}
		index = std::move(new_index);
		index_capacity = capacity;
		index_mask = mask;
	}

	uint32_t Insert(const string_t &value) {
		const auto hash = Hash(value);
		idx_t slot = hash & index_mask;
		while (true) {
			auto code = index[slot];
			if (code == 0) {
				break;
			}
			if (hashes[code] == hash && slots[code] == value) {
				return code;
			}
			slot = (slot + 1) & index_mask;
		}
		const auto code = UnsafeNumericCast<uint32_t>(slots.size());
		string_t stored = value.IsInlined() ? value : arena->AddString(value);
		slots.push_back(stored);
		hashes.push_back(hash);
		bytes += value.GetSize();
		index[slot] = code;
		if (slots.size() * 2 > index_capacity) {
			ResizeIndex(index_capacity * 2);
		}
		return code;
	}

	idx_t Footprint() const {
		return bytes + slots.capacity() * sizeof(string_t) + hashes.capacity() * sizeof(hash_t) +
		       index_capacity * sizeof(uint32_t) + translation_bytes;
	}

	Reservation reservation;
	buffer_ptr<VectorStringBuffer> arena;
	vector<string_t> slots;
	vector<hash_t> hashes;
	unsafe_unique_array<uint32_t> index;
	idx_t index_capacity = 0;
	idx_t index_mask = 0;
	idx_t bytes = 0;
	idx_t translation_bytes = 0;
	vector<PendingTranslation> translations;
	//! the string heaps of a parallel build (its partitions' arenas); the published child references each
	vector<buffer_ptr<VectorBuffer>> heaps;
};

static hash_t NullHash() {
	Vector null_vector(LogicalType::VARCHAR, 1);
	FlatVector::SetNull(null_vector, 0, true);
	Vector hash_vector(LogicalType::HASH, 1);
	VectorOperations::Hash(null_vector, hash_vector, 1);
	return FlatVector::GetData<hash_t>(hash_vector)[0];
}

//! The column's DICT_FSST segments a build translates (persistent, keyed), in collection order: BuildColumn's own walk
static void CollectBuildSegments(RowGroupCollection &row_groups, idx_t storage_index,
                                 vector<reference<ColumnSegment>> &segments) {
	auto tree = row_groups.GetRowGroups();
	for (auto row_group_node = tree->GetRootSegment(); row_group_node;
	     row_group_node = tree->GetNextSegment(*row_group_node)) {
		auto &column = row_group_node->GetNode().GetRawColumnData(storage_index);
		auto &segment_tree = column.GetSegmentTree();
		for (auto segment_node = segment_tree.GetRootSegment(); segment_node;
		     segment_node = segment_tree.GetNextSegment(*segment_node)) {
			auto &segment = segment_node->GetNode();
			if (segment.GetCompressionFunction().type != CompressionType::COMPRESSION_DICT_FSST ||
			    segment.segment_type != ColumnSegmentType::PERSISTENT || segment.GetDictionaryCacheKey().empty()) {
				continue;
			}
			segments.push_back(segment);
		}
	}
}

static BuildOutcome BuildColumn(ClientContext &context, RowGroupCollection &row_groups, ColumnDictionary &entry,
                                Builder &builder) {
	auto &buffer_manager = BufferManager::GetBufferManager(entry.db);
	auto tree = row_groups.GetRowGroups();
	for (auto row_group_node = tree->GetRootSegment(); row_group_node;
	     row_group_node = tree->GetNextSegment(*row_group_node)) {
		auto &row_group = row_group_node->GetNode();
		auto &column = row_group.GetRawColumnData(entry.storage_index);
		auto &segments = column.GetSegmentTree();
		for (auto segment_node = segments.GetRootSegment(); segment_node;
		     segment_node = segments.GetNextSegment(*segment_node)) {
			if (context.interrupted) {
				return BuildOutcome::INTERRUPTED;
			}
			auto &segment = segment_node->GetNode();
			if (segment.GetCompressionFunction().type != CompressionType::COMPRESSION_DICT_FSST ||
			    segment.segment_type != ColumnSegmentType::PERSISTENT || segment.GetDictionaryCacheKey().empty()) {
				continue;
			}
			dict_fsst::CompressedStringScanState scan_state(segment, buffer_manager.Pin(segment.block));
			scan_state.Initialize(true);
			if (scan_state.mode == dict_fsst::DictFSSTMode::FSST_ONLY) {
				continue;
			}
			scan_state.EnsureDictionary();
			if (!scan_state.dictionary) {
				continue;
			}
			const idx_t dict_count = scan_state.dict_count;
			auto strings = FlatVector::GetData<string_t>(scan_state.dictionary->data);
			auto codes = make_unsafe_uniq_array<uint32_t>(dict_count);
			codes[0] = 0;
			for (idx_t local = 1; local < dict_count; local++) {
				codes[local] = builder.Insert(strings[local]);
			}
			builder.translation_bytes += dict_count * sizeof(uint32_t);
			builder.translations.push_back(
			    PendingTranslation {segment.GetDictionaryCacheKey(), std::move(codes), dict_count});
			builder.reservation.Grow(builder.Footprint());
		}
	}
	if (builder.translations.empty()) {
		// no DICT_FSST segment: nothing to translate, so no dictionary is published (the column keeps its per-segment
		// vectors for the process; its keys are never typed)
		return BuildOutcome::REFUSED_EMPTY;
	}
	return BuildOutcome::PUBLISHED;
}

static void Publish(ColumnDictionary &entry, Builder &builder, const shared_ptr<ColumnDictionary> &shared_entry) {
	const idx_t count = builder.slots.size();
	// the child and its cached hashes are written once, here, before the release store below
	builder.reservation.Grow(builder.Footprint() + count * (sizeof(string_t) + sizeof(hash_t)));
	auto child = DictionaryVector::CreateReusableDictionary(LogicalType::VARCHAR, count);
	child->id = "dict_global-" + UUID::ToString(UUID::GenerateRandomUUID());
	auto child_strings = FlatVector::GetData<string_t>(child->data);
	memcpy(child_strings, builder.slots.data(), count * sizeof(string_t));
	FlatVector::Validity(child->data).SetInvalid(0);
	StringVector::AddBuffer(child->data, builder.arena);
	for (auto &heap : builder.heaps) {
		StringVector::AddBuffer(child->data, heap);
	}
	builder.heaps.clear();
	child->cached_hashes.Initialize(false, count);
	auto child_hashes = FlatVector::GetData<hash_t>(child->cached_hashes);
	memcpy(child_hashes, builder.hashes.data(), count * sizeof(hash_t));
	child_hashes[0] = NullHash();

	entry.child = std::move(child);
	entry.id = entry.child->id;
	entry.count = count;
	entry.bytes = builder.bytes;
	entry.index = std::move(builder.index);
	entry.index_mask = builder.index_mask;
	// the slots and hashes now live in the child: the builder's copies are released, the reservation kept
	vector<string_t>().swap(builder.slots);
	vector<hash_t>().swap(builder.hashes);

	auto &cache = entry.db.GetObjectCache();
	for (auto &translation : builder.translations) {
		cache.Put(TranslationKey(translation.segment_key),
		          make_shared_ptr<SegmentTranslation>(shared_entry, std::move(translation.codes), translation.count));
	}
	builder.translations.clear();
	// the build's reservation, already in the shared account, becomes the entry's
	entry.reserved = builder.reservation.Release();
	any_published.store(true, std::memory_order_release);
	entry.state.store(static_cast<uint8_t>(ColumnState::PUBLISHED), std::memory_order_release);
	publication_version.fetch_add(1, std::memory_order_release);
	if (XlateSegCacheEnabled()) {
		translation_generation.fetch_add(1, std::memory_order_acq_rel);
	}
}


//===--------------------------------------------------------------------===//
// The build in parallel
//===--------------------------------------------------------------------===//
static constexpr idx_t PARALLEL_PARTITION_BITS = 10;
static constexpr idx_t PARALLEL_PARTITIONS = 1ULL << PARALLEL_PARTITION_BITS;
//! the segments one dedup task decodes: below two tasks' worth the build stays on the calling thread (the standard form)
static constexpr idx_t PARALLEL_SEGMENTS_PER_TASK = 8;

static unique_ptr<TaskExecutor> MakeExecutor(ClientContext &context) {
	return make_uniq<TaskExecutor>(context);
}

//! One hash partition of the column's distinct strings, filled concurrently under its own lock; its arena becomes one of
//! the published child's string heaps
struct BuildPartition {
	mutex lock;
	unsafe_unique_array<uint32_t> table;
	idx_t capacity = 0;
	vector<string_t> strings;
	vector<hash_t> hashes;
	buffer_ptr<VectorStringBuffer> arena;
	idx_t bytes = 0;

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

	//! The string's id within the partition (added when new); `added` grows by the memory a new string takes
	uint32_t Insert(const string_t &value, hash_t hash, idx_t &added) {
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
				return id;
			}
			slot = (slot + 1) & mask;
		}
		const auto id = UnsafeNumericCast<uint32_t>(strings.size());
		const idx_t strings_capacity = strings.capacity();
		const idx_t hashes_capacity = hashes.capacity();
		strings.push_back(value.IsInlined() ? value : arena->AddString(value));
		hashes.push_back(hash);
		bytes += value.GetSize();
		table[slot] = id + 1;
		// the memory this insert took: the string's arena bytes, and the vectors' growth when they reallocated
		added += value.IsInlined() ? 0 : value.GetSize();
		added += (strings.capacity() - strings_capacity) * sizeof(string_t) + (hashes.capacity() - hashes_capacity) * sizeof(hash_t);
		if (strings.size() * 2 > capacity) {
			added += capacity * 2 * sizeof(uint32_t);
			Resize(capacity * 2);
		}
		return id;
	}
};

//! One segment's local codes as (partition << 32 | id) until the partitions are numbered
struct ParallelSegment {
	uint32_t dict_count = 0;
	unsafe_unique_array<uint64_t> packed;
};

struct ParallelBuildState {
	ParallelBuildState(ClientContext &context_p, BufferManager &buffer_manager_p, Builder &builder_p)
	    : context(context_p), buffer_manager(buffer_manager_p), builder(builder_p) {
	}
	ClientContext &context;
	BufferManager &buffer_manager;
	Builder &builder;
	vector<reference<ColumnSegment>> segments;
	vector<ParallelSegment> work;
	unsafe_unique_array<BuildPartition> partitions;
	vector<uint64_t> base;
	idx_t index_mask = 0;
	atomic<bool> refused {false};
	atomic<bool> interrupted {false};
	//! the build's one reservation (the shared global-dictionary account through Reservation), grown by the tasks under a
	//! lock
	mutex reservation_lock;
	idx_t footprint = 0;

	void Account(idx_t added) {
		lock_guard<mutex> guard(reservation_lock);
		footprint += added;
		builder.reservation.Grow(footprint);
	}
};

class ParallelDedupTask : public BaseExecutorTask {
public:
	ParallelDedupTask(TaskExecutor &executor, ParallelBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t k = begin; k < end; k++) {
			if (state.refused.load(std::memory_order_relaxed) || state.interrupted.load(std::memory_order_relaxed)) {
				return;
			}
			if (state.context.interrupted) {
				state.interrupted = true;
				return;
			}
			auto &segment = state.segments[k].get();
			auto &work = state.work[k];
			dict_fsst::CompressedStringScanState scan_state(segment, state.buffer_manager.Pin(segment.block));
			scan_state.Initialize(false);
			if (scan_state.mode == dict_fsst::DictFSSTMode::FSST_ONLY) {
				continue;
			}
			const idx_t dict_count = scan_state.dict_count;
			idx_t added = dict_count * sizeof(uint64_t);
			try {
				state.Account(added);
				added = 0;
				work.packed = make_unsafe_uniq_array_uninitialized<uint64_t>(dict_count);
			} catch (OutOfMemoryException &) {
				state.refused = true;
				return;
			} catch (std::bad_alloc &) {
				state.refused = true;
				return;
			}
			work.packed[0] = 0;
			Vector decoded(LogicalType::VARCHAR, 1);
			uint32_t offset = 0;
			for (idx_t local = 0; local < dict_count; local++) {
				const auto length = scan_state.string_lengths[local];
				if (local > 0) {
					const auto value = scan_state.FetchEntry(decoded, offset, local);
					const auto hash = Hash(value);
					const auto partition = hash >> (64 - PARALLEL_PARTITION_BITS);
					const auto id = state.partitions[partition].Insert(value, hash, added);
					work.packed[local] = (static_cast<uint64_t>(partition) << 32) | id;
				}
				offset += length;
			}
			work.dict_count = UnsafeNumericCast<uint32_t>(dict_count);
			try {
				state.Account(added);
			} catch (OutOfMemoryException &) {
				state.refused = true;
				return;
			}
		}
	}
	string TaskType() const override {
		return "GlobalDictionaryParallelDedupTask";
	}

private:
	ParallelBuildState &state;
	idx_t begin;
	idx_t end;
};

//! The child's slots and hashes from a range of partitions, then those partitions' tables and vectors released
class ParallelSlotsTask : public BaseExecutorTask {
public:
	ParallelSlotsTask(TaskExecutor &executor, ParallelBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t p = begin; p < end; p++) {
			auto &partition = state.partitions[p];
			const idx_t n = partition.strings.size();
			if (n) {
				memcpy(state.builder.slots.data() + state.base[p], partition.strings.data(), n * sizeof(string_t));
				memcpy(state.builder.hashes.data() + state.base[p], partition.hashes.data(), n * sizeof(hash_t));
			}
			partition.table.reset();
			vector<string_t>().swap(partition.strings);
			vector<hash_t>().swap(partition.hashes);
		}
	}
	string TaskType() const override {
		return "GlobalDictionaryParallelSlotsTask";
	}

private:
	ParallelBuildState &state;
	idx_t begin;
	idx_t end;
};

//! Each segment's translation (local code -> global code) from its packed codes
class ParallelTranslateTask : public BaseExecutorTask {
public:
	ParallelTranslateTask(TaskExecutor &executor, ParallelBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		for (idx_t k = begin; k < end; k++) {
			auto &work = state.work[k];
			if (!work.dict_count) {
				continue;
			}
			auto &segment = state.segments[k].get();
			auto codes = make_unsafe_uniq_array_uninitialized<uint32_t>(work.dict_count);
			codes[0] = 0;
			for (idx_t local = 1; local < work.dict_count; local++) {
				const auto packed = work.packed[local];
				codes[local] = UnsafeNumericCast<uint32_t>(state.base[packed >> 32] + (packed & 0xFFFFFFFFULL));
			}
			work.packed.reset();
			state.builder.translations[k] =
			    PendingTranslation {segment.GetDictionaryCacheKey(), std::move(codes), work.dict_count};
		}
	}
	string TaskType() const override {
		return "GlobalDictionaryParallelTranslateTask";
	}

private:
	ParallelBuildState &state;
	idx_t begin;
	idx_t end;
};

//! The lookup index over the distinct strings: every code's string is distinct, so an insert only looks for an empty slot
class ParallelIndexTask : public BaseExecutorTask {
public:
	ParallelIndexTask(TaskExecutor &executor, ParallelBuildState &state, idx_t begin, idx_t end)
	    : BaseExecutorTask(executor), state(state), begin(begin), end(end) {
	}
	void ExecuteTask() override {
		auto index = state.builder.index.get();
		auto hashes = state.builder.hashes.data();
		for (idx_t code = begin; code < end; code++) {
			idx_t slot = hashes[code] & state.index_mask;
			while (!__sync_bool_compare_and_swap(&index[slot], 0U, UnsafeNumericCast<uint32_t>(code))) {
				slot = (slot + 1) & state.index_mask;
			}
		}
	}
	string TaskType() const override {
		return "GlobalDictionaryParallelIndexTask";
	}

private:
	ParallelBuildState &state;
	idx_t begin;
	idx_t end;
};

//! The column's dictionary built as BuildColumn builds it - the same segments, the same distinct strings and hashes, the
//! same translations, published by the same Publish - with its work spread over the scheduler's threads: the segments
//! are decoded (read, when not buffered) and hashed by tasks of PARALLEL_SEGMENTS_PER_TASK segments into 2^10 hash
//! partitions, each under its own lock. The codes are numbered by partition (slot 0 NULL, then each partition's strings in
//! order), not by first occurrence. Every allocation is reserved in the build's one Reservation (the shared account), with
//! the standard pressure refusal
static BuildOutcome BuildColumnParallel(ClientContext &context, RowGroupCollection &row_groups, ColumnDictionary &entry,
                                        Builder &builder) {
	auto &buffer_manager = BufferManager::GetBufferManager(entry.db);
	ParallelBuildState state(context, buffer_manager, builder);
	CollectBuildSegments(row_groups, entry.storage_index, state.segments);
	const idx_t segment_count = state.segments.size();
	const idx_t threads = MaxValue<int32_t>(1, TaskScheduler::GetScheduler(context).NumberOfThreads());
	const bool parallel = segment_count > PARALLEL_SEGMENTS_PER_TASK && threads > 1;
	if (!parallel) {
		// one task's worth of segments, or one thread: nothing to spread - the calling thread builds as the standard build
		return BuildColumn(context, row_groups, entry, builder);
	}
	state.work.resize(segment_count);
	state.Account(PARALLEL_PARTITIONS * sizeof(BuildPartition));
	state.partitions = make_unsafe_uniq_array<BuildPartition>(PARALLEL_PARTITIONS);
	{
		auto executor = MakeExecutor(context);
		for (idx_t begin = 0; begin < segment_count; begin += PARALLEL_SEGMENTS_PER_TASK) {
			executor->ScheduleTask(make_uniq<ParallelDedupTask>(*executor, state, begin,
			                                                     MinValue(segment_count, begin + PARALLEL_SEGMENTS_PER_TASK)));
		}
		executor->WorkOnTasks();
	}
	if (state.interrupted) {
		return BuildOutcome::INTERRUPTED;
	}
	if (state.refused) {
		throw OutOfMemoryException("Global dictionary: the column dictionary's parallel build refused under memory pressure");
	}
	idx_t translated = 0;
	idx_t translation_entries = 0;
	for (auto &work : state.work) {
		translated += work.dict_count ? 1 : 0;
		translation_entries += work.dict_count;
	}
	if (translated == 0) {
		return BuildOutcome::REFUSED_EMPTY;
	}
	// the code space: slot 0 (NULL), then each partition's strings in partition order
	state.base.resize(PARALLEL_PARTITIONS + 1);
	state.base[0] = 1;
	idx_t bytes = 0;
	for (idx_t p = 0; p < PARALLEL_PARTITIONS; p++) {
		state.base[p + 1] = state.base[p] + state.partitions[p].strings.size();
		bytes += state.partitions[p].bytes;
	}
	const idx_t count = state.base[PARALLEL_PARTITIONS];
	if (count > static_cast<idx_t>(NumericLimits<int32_t>::Maximum())) {
		throw OutOfMemoryException("Global dictionary: the column dictionary's code space exceeds 2^31 entries");
	}
	idx_t index_capacity = 1ULL << 16;
	while (count * 2 > index_capacity) {
		index_capacity *= 2;
	}
	// the published structures, reserved before they are allocated: the child's slots and hashes, the index, the
	// translations (the partitions' vectors are still held while the slots are filled)
	builder.bytes = bytes;
	builder.translation_bytes = translation_entries * sizeof(uint32_t);
	state.Account(count * (sizeof(string_t) + sizeof(hash_t)) + index_capacity * sizeof(uint32_t) +
	              builder.translation_bytes);
	builder.slots.resize(count);
	builder.hashes.resize(count);
	builder.slots[0] = string_t(nullptr, 0);
	builder.hashes[0] = 0;
	{
		auto executor = MakeExecutor(context);
		const idx_t step = MaxValue<idx_t>(1, PARALLEL_PARTITIONS / threads);
		for (idx_t begin = 0; begin < PARALLEL_PARTITIONS; begin += step) {
			executor->ScheduleTask(
			    make_uniq<ParallelSlotsTask>(*executor, state, begin, MinValue(PARALLEL_PARTITIONS, begin + step)));
		}
		executor->WorkOnTasks();
	}
	builder.translations.resize(segment_count);
	{
		auto executor = MakeExecutor(context);
		for (idx_t begin = 0; begin < segment_count; begin += PARALLEL_SEGMENTS_PER_TASK) {
			executor->ScheduleTask(make_uniq<ParallelTranslateTask>(
			    *executor, state, begin, MinValue(segment_count, begin + PARALLEL_SEGMENTS_PER_TASK)));
		}
		executor->WorkOnTasks();
	}
	// FSST_ONLY segments have no translation
	idx_t kept = 0;
	for (idx_t k = 0; k < segment_count; k++) {
		if (builder.translations[k].codes) {
			if (kept != k) {
				builder.translations[kept] = std::move(builder.translations[k]);
			}
			kept++;
		}
	}
	builder.translations.resize(kept);
	builder.index = make_unsafe_uniq_array<uint32_t>(index_capacity);
	memset(builder.index.get(), 0, index_capacity * sizeof(uint32_t));
	builder.index_capacity = index_capacity;
	builder.index_mask = index_capacity - 1;
	state.index_mask = builder.index_mask;
	{
		auto executor = MakeExecutor(context);
		const idx_t step = MaxValue<idx_t>(1ULL << 16, count / (threads * 4) + 1);
		for (idx_t begin = 1; begin < count; begin += step) {
			executor->ScheduleTask(make_uniq<ParallelIndexTask>(*executor, state, begin, MinValue(count, begin + step)));
		}
		executor->WorkOnTasks();
	}
	// the partitions' arenas hold the strings the slots point at: the child references each
	for (idx_t p = 0; p < PARALLEL_PARTITIONS; p++) {
		if (state.partitions[p].arena) {
			builder.heaps.push_back(std::move(state.partitions[p].arena));
		}
	}
	state.partitions.reset();
	// the transient state (the partitions' tables and vectors, the packed codes) is gone: the reservation keeps what the
	// published dictionary holds (Footprint(), as the standard build's), so the budget left to other builds is the standard one
	builder.reservation.ShrinkTo(builder.Footprint());
	return BuildOutcome::PUBLISHED;
}

} // namespace

void EnsureBuilt(ClientContext &context, DataTable &table, const vector<idx_t> &storage_ids) {
	if (!DictGlobalEnabled() || storage_ids.empty()) {
		return;
	}
	// only a scan whose planner marked a column (a grouping-key or DISTINCT consumer above
	// it) builds that column; an unmarked scan builds nothing
	auto publication = ThreadPublication();
	if (!publication) {
		return;
	}
	auto &info_ptr = table.GetDataTableInfo();
	auto registry_entry = TableEntry(info_ptr, true);
	auto &registry = *registry_entry;
	auto &columns = table.Columns();
	auto &collection = *table.GetRowGroupCollection();
	for (auto storage_index : storage_ids) {
		optional_ptr<const ColumnDefinition> definition;
		for (auto &column : columns) {
			if (column.StorageOid() == storage_index) {
				definition = &column;
				break;
			}
		}
		// VARCHAR columns only: the published child is a VARCHAR vector, so a BLOB or BIT column (also stored as
		// strings) is not given one and scans as it would without a published dictionary
		if (!definition || definition->Type().id() != LogicalTypeId::VARCHAR) {
			continue;
		}
		auto marked = publication->Find(storage_index);
		if (!marked) {
			continue;
		}
		if (marked->gated) {
			continue;
		}
		shared_ptr<ColumnDictionary> entry;
		{
			lock_guard<mutex> guard(registry.lock);
			const auto epoch = registry.epoch.load();
			auto existing = registry.columns.find(storage_index);
			if (existing != registry.columns.end() && existing->second->epoch == epoch &&
			    existing->second->collection == &collection) {
				entry = existing->second;
			} else {
				entry = make_shared_ptr<ColumnDictionary>(table.db.GetDatabase(), collection, storage_index, epoch,
				                                          definition->Name());
				registry.columns[storage_index] = entry;
			}
		}
		if (EstimateGateEnabled() && entry->GetState() == ColumnState::UNBUILT) {
			// the estimate refusal: a build that its own estimate says cannot fit is not started - no reservation is
			// taken and nothing is decoded; the entry stays unbuilt and this execution's scan takes the general path (as a
			// refused build's does), so a query never waits on, or runs short of memory for, a build that would be refused
			const idx_t limit = BufferManager::GetBufferManager(entry->db).GetMaxMemory();
			if (total_reserved.load() + marked->estimated_reserve > Budget(limit)) {
				continue;
			}
		}
		auto expected = static_cast<uint8_t>(ColumnState::UNBUILT);
		if (!entry->state.compare_exchange_strong(expected, static_cast<uint8_t>(ColumnState::BUILDING),
		                                          std::memory_order_acq_rel)) {
			continue;
		}
		BuildOutcome outcome;
		{
			unique_ptr<Builder> builder;
			try {
				builder = make_uniq<Builder>(BufferManager::GetBufferManager(entry->db));
				outcome = ParallelBuildEnabled() ? BuildColumnParallel(context, collection, *entry, *builder)
				                               : BuildColumn(context, collection, *entry, *builder);
				if (outcome == BuildOutcome::PUBLISHED) {
					Publish(*entry, *builder, entry);
				}
			} catch (OutOfMemoryException &) {
				outcome = BuildOutcome::REFUSED_MEMORY;
			} catch (std::bad_alloc &) {
				outcome = BuildOutcome::REFUSED_MEMORY;
			} catch (...) {
				// any other failure (an I/O error, an interrupt raised inside the decode) leaves the column unbuilt for a
				// later scan and surfaces as the query's own error, as the standard scan would
				entry->state.store(static_cast<uint8_t>(ColumnState::UNBUILT), std::memory_order_release);
				throw;
			}
			// a refused or interrupted build releases its partial state and reservation here
		}
		if (outcome == BuildOutcome::REFUSED_MEMORY || outcome == BuildOutcome::REFUSED_EMPTY) {
			entry->state.store(static_cast<uint8_t>(ColumnState::REFUSED), std::memory_order_release);
		} else if (outcome == BuildOutcome::INTERRUPTED) {
			entry->state.store(static_cast<uint8_t>(ColumnState::UNBUILT), std::memory_order_release);
		}
	}
}

static constexpr const char *PUBLISHED_ID_PREFIX = "dict_global-";
static constexpr idx_t PUBLISHED_ID_PREFIX_LENGTH = 12;

bool IsPublishedVector(const Vector &vector) {
	if (vector.GetVectorType() != VectorType::DICTIONARY_VECTOR ||
	    vector.GetType().InternalType() != PhysicalType::VARCHAR) {
		return false;
	}
	auto &id = DictionaryVector::DictionaryId(vector);
	return id.size() > PUBLISHED_ID_PREFIX_LENGTH && id.compare(0, PUBLISHED_ID_PREFIX_LENGTH, PUBLISHED_ID_PREFIX) == 0;
}

idx_t &ThreadBorrowColumn() {
	thread_local idx_t column = DConstants::INVALID_INDEX;
	return column;
}

//===--------------------------------------------------------------------===//
// Code keys
//===--------------------------------------------------------------------===//
void CodeKeys::ConvertInput(DataChunk &input, DataChunk &converted, vector<Vector> &code_vectors) const {
	const idx_t count = input.size();
	for (idx_t col = 0; col < input.ColumnCount(); col++) {
		bool typed = false;
		for (auto &key : keys) {
			typed = typed || key.chunk_index == col;
		}
		if (!typed) {
			converted.data[col].Reference(input.data[col]);
		}
	}
	for (idx_t k = 0; k < keys.size(); k++) {
		auto &key = keys[k];
		auto &source = input.data[key.chunk_index];
		auto &target = code_vectors[k];
		target.SetVectorType(VectorType::FLAT_VECTOR);
		auto codes = FlatVector::GetData<int32_t>(target);
		if (source.GetVectorType() == VectorType::DICTIONARY_VECTOR &&
		    &DictionaryVector::Child(source) == &key.dict->child->data) {
			auto &sel = DictionaryVector::SelVector(source);
			for (idx_t i = 0; i < count; i++) {
				codes[i] = UnsafeNumericCast<int32_t>(sel.get_index(i));
			}
		} else {
			// a vector not over the recorded child: every row looked up in the published dictionary (a string the
			// dictionary does not hold takes an overflow code above the published count)
			UnifiedVectorFormat format;
			source.ToUnifiedFormat(count, format);
			auto strings = UnifiedVectorFormat::GetData<string_t>(format);
			for (idx_t i = 0; i < count; i++) {
				auto idx = format.sel->get_index(i);
				if (!format.validity.RowIsValid(idx)) {
					codes[i] = 0;
					continue;
				}
				auto &value = strings[idx];
				auto code = key.dict->Lookup(value, Hash(value));
				if (code == ColumnDictionary::INVALID_CODE) {
					auto &overflow_entry = *overflow[k];
					lock_guard<mutex> guard(overflow_entry.lock);
					auto text = value.GetString();
					auto found = overflow_entry.codes.find(text);
					if (found == overflow_entry.codes.end()) {
						code = UnsafeNumericCast<uint32_t>(key.dict->count + overflow_entry.strings.size());
						overflow_entry.codes.emplace(text, code);
						overflow_entry.strings.push_back(std::move(text));
					} else {
						code = found->second;
					}
				}
				codes[i] = UnsafeNumericCast<int32_t>(code);
			}
		}
		converted.data[key.chunk_index].Reference(target);
	}
	converted.SetCardinality(count);
}

void CodeKeys::ConvertOutput(DataChunk &internal, DataChunk &output) const {
	const idx_t count = internal.size();
	for (idx_t col = 0; col < output.ColumnCount(); col++) {
		optional_ptr<const CodeKey> typed;
		idx_t key_idx = 0;
		for (idx_t k = 0; k < keys.size(); k++) {
			if (keys[k].group_index == col) {
				typed = &keys[k];
				key_idx = k;
			}
		}
		if (!typed) {
			output.data[col].Reference(internal.data[col]);
			continue;
		}
		auto &dict = *typed->dict;
		UnifiedVectorFormat format;
		internal.data[col].ToUnifiedFormat(count, format);
		auto codes = UnifiedVectorFormat::GetData<int32_t>(format);
		bool overflowed = false;
		SelectionVector sel(MaxValue<idx_t>(count, 1));
		for (idx_t i = 0; i < count; i++) {
			auto code = UnsafeNumericCast<idx_t>(codes[format.sel->get_index(i)]);
			overflowed = overflowed || code >= dict.count;
			sel.set_index(i, code);
		}
		if (!overflowed) {
			output.data[col].Dictionary(dict.child, sel);
			continue;
		}
		// an overflow code: a flat string vector
		auto &target = output.data[col];
		target.SetVectorType(VectorType::FLAT_VECTOR);
		auto strings = FlatVector::GetData<string_t>(target);
		auto &validity = FlatVector::Validity(target);
		auto child_strings = FlatVector::GetData<string_t>(dict.child->data);
		auto &overflow_entry = *overflow[key_idx];
		lock_guard<mutex> guard(overflow_entry.lock);
		for (idx_t i = 0; i < count; i++) {
			auto code = sel.get_index(i);
			if (code == 0) {
				validity.SetInvalid(i);
				continue;
			}
			if (code < dict.count) {
				strings[i] = StringVector::AddStringOrBlob(target, child_strings[code]);
			} else {
				strings[i] = StringVector::AddStringOrBlob(target, overflow_entry.strings[code - dict.count]);
			}
		}
	}
	output.SetCardinality(count);
}

namespace {
struct Resolved {
	bool found = false;
	string name;
	LogicalType type;
	idx_t storage_index = 0;
};

//! Resolve column `index` of the operator's input through PROJECTION* / FILTER* (BOUND_REFs only) to a base-table
//! column of the scan
static Resolved ResolveColumn(const vector<reference<PhysicalOperator>> &chain, PhysicalTableScan &scan,
                              DuckTableEntry &table, idx_t index) {
	Resolved result;
	for (auto &operator_ref : chain) {
		auto &op = operator_ref.get();
		if (op.type == PhysicalOperatorType::FILTER) {
			continue;
		}
		auto &projection = op.Cast<PhysicalProjection>();
		if (index >= projection.select_list.size()) {
			return result;
		}
		auto &expr = *projection.select_list[index];
		if (expr.GetExpressionType() != ExpressionType::BOUND_REF) {
			return result;
		}
		index = expr.Cast<BoundReferenceExpression>().index;
	}
	idx_t scan_index = index;
	if (!scan.projection_ids.empty()) {
		if (index >= scan.projection_ids.size()) {
			return result;
		}
		scan_index = scan.projection_ids[index];
	}
	if (scan_index >= scan.column_ids.size()) {
		return result;
	}
	auto &column_index = scan.column_ids[scan_index];
	if (column_index.IsRowIdColumn() || column_index.IsVirtualColumn() || column_index.HasChildren() ||
	    !column_index.HasPrimaryIndex()) {
		return result;
	}
	auto &column = table.GetColumns().GetColumn(LogicalIndex(column_index.GetPrimaryIndex()));
	if (column.Generated()) {
		return result;
	}
	result.found = true;
	result.name = column.Name();
	result.type = column.Type();
	result.storage_index = column.StorageOid();
	return result;
}
} // namespace

unique_ptr<CodeKeys> PlanCodeKeys(ClientContext &context, PhysicalOperator &child,
                                  vector<unique_ptr<Expression>> &groups, const vector<LogicalType> &output_types,
                                  const vector<unique_ptr<Expression>> &aggregates, idx_t grouping_set_count) {
	if (!CodeKeysEnabled() || groups.empty()) {
		return nullptr;
	}
	// the chain: PROJECTION* / FILTER* over one seq_scan of a DuckDB table
	vector<reference<PhysicalOperator>> chain;
	reference<PhysicalOperator> current(child);
	while (current.get().type == PhysicalOperatorType::PROJECTION ||
	       current.get().type == PhysicalOperatorType::FILTER) {
		if (current.get().children.size() != 1) {
			return nullptr;
		}
		chain.push_back(current);
		current = current.get().children[0];
	}
	if (current.get().type != PhysicalOperatorType::TABLE_SCAN) {
		return nullptr;
	}
	auto &scan = current.get().Cast<PhysicalTableScan>();
	if (scan.function.name != "seq_scan" || !scan.bind_data) {
		return nullptr;
	}
	auto &table_entry = scan.bind_data->Cast<TableScanBindData>().table;
	if (!table_entry.IsDuckTable()) {
		return nullptr;
	}
	auto &duck_table = table_entry.Cast<DuckTableEntry>();

	// the chunk indexes the aggregates (and their filters) read
	vector<idx_t> read_indexes;
	for (auto &aggregate : aggregates) {
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		for (auto &aggr_child : aggr.children) {
			if (aggr_child->GetExpressionType() == ExpressionType::BOUND_REF) {
				read_indexes.push_back(aggr_child->Cast<BoundReferenceExpression>().index);
			}
		}
		if (aggr.filter && aggr.filter->GetExpressionType() == ExpressionType::BOUND_REF) {
			read_indexes.push_back(aggr.filter->Cast<BoundReferenceExpression>().index);
		}
	}

	auto result = make_uniq<CodeKeys>();
	for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
		auto &group = *groups[group_idx];
		if (group.GetExpressionType() != ExpressionType::BOUND_REF ||
		    group.return_type.InternalType() != PhysicalType::VARCHAR) {
			continue;
		}
		auto chunk_index = group.Cast<BoundReferenceExpression>().index;
		auto resolved = ResolveColumn(chain, scan, duck_table, chunk_index);
		if (!resolved.found || resolved.type.InternalType() != PhysicalType::VARCHAR) {
			continue;
		}
		shared_ptr<ColumnDictionary> dict;
		if (StringType::GetCollation(resolved.type).empty() && StringType::GetCollation(group.return_type).empty() &&
		    grouping_set_count <= 1 &&
		    std::find(read_indexes.begin(), read_indexes.end(), chunk_index) == read_indexes.end() &&
		    ScanPublicationOf(scan.bind_data.get()) &&
		    ScanPublicationOf(scan.bind_data.get())->Publishes(resolved.storage_index)) {
			// the scan publishes the column for this plan (marked, and not refused by the memory gate)
			dict = Published(duck_table.GetStorage(), resolved.storage_index);
		}
		if (!dict) {
			continue;
		}
		CodeKey key;
		key.group_index = group_idx;
		key.chunk_index = chunk_index;
		key.dict = std::move(dict);
		if (!result->column_names.empty()) {
			result->column_names += "\n";
		}
		result->column_names += resolved.name;
		result->keys.push_back(std::move(key));
	}
	if (result->keys.empty()) {
		return nullptr;
	}
	// the wide guard: with a packed group width (plan_aggregate's key_bytes) above the fused kernel's 12 bytes once these
	// keys are typed, the code key would land on the general hash table, where it is slower than the string key;
	// such keys stay strings (the 12-byte bound is a tuning threshold; see CLICKBENCH-FORK.md)
	if (WideGuardEnabled()) {
		idx_t key_bytes = 0;
		for (idx_t group_idx = 0; group_idx < groups.size(); group_idx++) {
			bool typed = false;
			for (auto &key : result->keys) {
				typed = typed || key.group_index == group_idx;
			}
			key_bytes += typed ? GetTypeIdSize(PhysicalType::INT32)
			                   : GetTypeIdSize(groups[group_idx]->return_type.InternalType());
		}
		if (key_bytes > FusedIntegerAggregate::MAXIMUM_KEY_BYTES) {
			return nullptr;
		}
	}
	// type the keys: the group BOUND_REFs, the operator's input and internal output types
	result->input_types = child.types;
	result->internal_output_types = output_types;
	for (auto &key : result->keys) {
		groups[key.group_index]->return_type = LogicalType::INTEGER;
		result->input_types[key.chunk_index] = LogicalType::INTEGER;
		result->internal_output_types[key.group_index] = LogicalType::INTEGER;
		result->overflow.push_back(make_uniq<CodeKeys::Overflow>());
	}
	return result;
}

//! The string-predicate build guard; off, such a scan is marked as any other
static bool S1GuardEnabled() {
	return kStringPredicateBuildGuard;
}

//! Whether the pushed filter holds an expression filter (contains, LIKE, any non-constant predicate) at any depth of a
//! conjunction or an optional wrapper
static bool HoldsExpressionFilter(const TableFilter &filter) {
	switch (filter.filter_type) {
	case TableFilterType::EXPRESSION_FILTER:
		return true;
	case TableFilterType::CONJUNCTION_AND: {
		for (auto &child_filter : filter.Cast<ConjunctionAndFilter>().child_filters) {
			if (HoldsExpressionFilter(*child_filter)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::CONJUNCTION_OR: {
		for (auto &child_filter : filter.Cast<ConjunctionOrFilter>().child_filters) {
			if (HoldsExpressionFilter(*child_filter)) {
				return true;
			}
		}
		return false;
	}
	case TableFilterType::OPTIONAL_FILTER: {
		auto &optional = filter.Cast<OptionalFilter>();
		return optional.child_filter && HoldsExpressionFilter(*optional.child_filter);
	}
	default:
		return false;
	}
}

//! The string-predicate build guard: the scan carries a pushed expression filter on a VARCHAR column - a string
//! predicate whose selectivity the estimate cannot see, so the scan's estimate can pass the admission share while few
//! rows survive (a tuning rule; see CLICKBENCH-FORK.md)
static bool ScanHasStringExpressionFilter(const PhysicalTableScan &scan, DuckTableEntry &table) {
	if (!scan.table_filters) {
		return false;
	}
	for (auto &entry : scan.table_filters->filters) {
		if (!HoldsExpressionFilter(*entry.second) || entry.first >= scan.column_ids.size()) {
			continue;
		}
		auto &column_index = scan.column_ids[entry.first];
		if (column_index.IsRowIdColumn() || column_index.IsVirtualColumn() || !column_index.HasPrimaryIndex()) {
			continue;
		}
		auto &column = table.GetColumns().GetColumn(LogicalIndex(column_index.GetPrimaryIndex()));
		if (column.Type().InternalType() == PhysicalType::VARCHAR) {
			return true;
		}
	}
	return false;
}

void MarkKeyConsumers(ClientContext &context, PhysicalOperator &child, const vector<unique_ptr<Expression>> &groups,
                      const vector<unique_ptr<Expression>> &aggregates) {
	if (!DictGlobalEnabled()) {
		return;
	}
	// the chain: PROJECTION* / FILTER* over one seq_scan of a DuckDB table (PlanCodeKeys's chain)
	vector<reference<PhysicalOperator>> chain;
	reference<PhysicalOperator> current(child);
	while (current.get().type == PhysicalOperatorType::PROJECTION ||
	       current.get().type == PhysicalOperatorType::FILTER) {
		if (current.get().children.size() != 1) {
			return;
		}
		chain.push_back(current);
		current = current.get().children[0];
	}
	if (current.get().type != PhysicalOperatorType::TABLE_SCAN) {
		return;
	}
	auto &scan = current.get().Cast<PhysicalTableScan>();
	if (scan.function.name != "seq_scan" || !scan.bind_data) {
		return;
	}
	auto &table_entry = scan.bind_data->Cast<TableScanBindData>().table;
	if (!table_entry.IsDuckTable()) {
		return;
	}
	auto &duck_table = table_entry.Cast<DuckTableEntry>();
	if (S1GuardEnabled() && ScanHasStringExpressionFilter(scan, duck_table)) {
		// the string-predicate build guard: the scan stays unmarked - no build, no emission over a global dictionary,
		// no code key, and no translation lookup in the ObjectCache at each DICT_FSST segment it initialises (the
		// filter columns' included)
		return;
	}
	// the chunk indexes the group items and the DISTINCT aggregates' arguments read
	vector<idx_t> indexes;
	for (auto &group : groups) {
		if (group->GetExpressionType() == ExpressionType::BOUND_REF &&
		    group->return_type.InternalType() == PhysicalType::VARCHAR) {
			indexes.push_back(group->Cast<BoundReferenceExpression>().index);
		}
	}
	for (auto &aggregate : aggregates) {
		if (aggregate->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			continue;
		}
		auto &aggr = aggregate->Cast<BoundAggregateExpression>();
		if (!aggr.IsDistinct()) {
			continue;
		}
		for (auto &aggr_child : aggr.children) {
			if (aggr_child->GetExpressionType() == ExpressionType::BOUND_REF &&
			    aggr_child->return_type.InternalType() == PhysicalType::VARCHAR) {
				indexes.push_back(aggr_child->Cast<BoundReferenceExpression>().index);
			}
		}
	}
	for (auto index : indexes) {
		auto resolved = ResolveColumn(chain, scan, duck_table, index);
		if (!resolved.found || resolved.type.InternalType() != PhysicalType::VARCHAR) {
			continue;
		}
		auto publishing = dynamic_cast<PublishingTableScanBindData *>(scan.bind_data.get());
		if (!publishing) {
			// the first mark: the scan's bind data becomes the publishing subclass (the base fields copied)
			auto replacement = make_uniq<PublishingTableScanBindData>(scan.bind_data->Cast<TableScanBindData>());
			publishing = replacement.get();
			scan.bind_data = std::move(replacement);
		}
		if (!publishing->publication) {
			publishing->publication = make_shared_ptr<ScanPublication>();
		}
		auto &publication = *publishing->publication;
		if (publication.Find(resolved.storage_index)) {
			continue;
		}
		ScanPublication::Column column;
		column.storage_index = resolved.storage_index;
		column.name = resolved.name;
		auto stats = duck_table.GetStatistics(context, StorageIndex(resolved.storage_index));
		column.approx_unique = stats ? stats->GetDistinctCount() : 0;
		column.estimated_reserve = column.approx_unique * ESTIMATED_BYTES_PER_UNIQUE;
		column.estimated_rows = scan.estimated_cardinality;
		column.table_rows = duck_table.GetStorage().GetTotalRows();
		// the admission share: a scan estimated below GATE_SHARE of the table's rows is refused the column for its
		// executions, whatever the column's size - the build reads the whole column, not the rows the scan reads
		// (GATE_SHARE is a tuning threshold; see CLICKBENCH-FORK.md)
		column.gated = static_cast<double>(column.estimated_rows) < GATE_SHARE * static_cast<double>(column.table_rows);
		publication.columns.push_back(std::move(column));
	}
}

namespace {
struct CodeKeysRegistry {
	mutex lock;
	atomic<idx_t> count {0};
	unordered_map<const void *, shared_ptr<CodeKeys>> operators;
};
CodeKeysRegistry &CodeKeysMap() {
	// never destroyed (as the table registry): its code keys hold dictionaries
	static auto registry = new CodeKeysRegistry();
	return *registry;
}
} // namespace

void RegisterCodeKeys(const void *op, shared_ptr<CodeKeys> keys) {
	auto &registry = CodeKeysMap();
	lock_guard<mutex> guard(registry.lock);
	registry.operators[op] = std::move(keys);
	registry.count = registry.operators.size();
}

shared_ptr<CodeKeys> FindCodeKeys(const void *op) {
	auto &registry = CodeKeysMap();
	if (registry.count.load(std::memory_order_relaxed) == 0) {
		return nullptr;
	}
	lock_guard<mutex> guard(registry.lock);
	auto found = registry.operators.find(op);
	return found == registry.operators.end() ? nullptr : found->second;
}

void ReleaseCodeKeys(const void *op) {
	auto &registry = CodeKeysMap();
	if (registry.count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	lock_guard<mutex> guard(registry.lock);
	registry.operators.erase(op);
	registry.count = registry.operators.size();
}

} // namespace dict_global
} // namespace duckdb
