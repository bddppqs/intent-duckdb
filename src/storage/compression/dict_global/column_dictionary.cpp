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
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/storage/compression/dict_fsst/split_segment.hpp"
#include "duckdb/storage/compression/dict_global/persisted_translation.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

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

//! PlanCodeKeys refuses the code keys of an aggregate whose packed group width with them typed passes the fused kernel's
//! key bytes (WideGuardKeyBytes); off, every width is typed
static bool WideGuardEnabled() {
	return kGlobalDictionaryWideKeyGuard;
}

//! The wide guard's limit follows the fused kernel's grouped class: 16 key bytes when it takes three keys
//! (kFusedSixteenByteKeys), 12 otherwise
static idx_t WideGuardKeyBytes() {
	return kFusedSixteenByteKeys ? FusedIntegerAggregate::MAXIMUM_KEY_BYTES : 12;
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
// Columns read through stored translations
static constexpr const char *CODES_ONLY_ID_PREFIX = "dict_global_codes-";

namespace {
struct CodesOnlyRegistry {
	mutex lock;
	unordered_map<string, weak_ptr<PersistedTranslations>> ids;
};
CodesOnlyRegistry &CodesOnlyIds() {
	static auto registry = new CodesOnlyRegistry();
	return *registry;
}
} // namespace

shared_ptr<PersistedTranslations> CodesOnlyTranslationsOf(const string &child_id) {
	if (child_id.compare(0, strlen(CODES_ONLY_ID_PREFIX), CODES_ONLY_ID_PREFIX) != 0) {
		return nullptr;
	}
	auto &registry = CodesOnlyIds();
	lock_guard<mutex> guard(registry.lock);
	auto found = registry.ids.find(child_id);
	return found == registry.ids.end() ? nullptr : found->second.lock();
}

//! Link every segment of the column to its stored translation entry: true when every segment of every row group is a
//! DICT_FSST segment the translations hold, the column has no updates, and its rows are the table's (a vector that
//! straddles two segments is emitted over the translations by TryScanGlobalDictionary)
static bool LinkPersisted(DataTable &table, idx_t storage_index, const shared_ptr<PersistedTranslations> &translations,
                          idx_t &rows) {
	auto &collection = *table.GetRowGroupCollection();
	auto tree = collection.GetRowGroups();
	auto &entries = translations->Entries();
	rows = 0;
	idx_t linked = 0;
	for (auto node = tree->GetRootSegment(); node; node = tree->GetNextSegment(*node)) {
		auto &column = node->GetNode().GetRawColumnData(storage_index);
		if (column.HasUpdates()) {
			return false;
		}
		auto &segments = column.GetSegmentTree();
		for (auto segment_node = segments.GetRootSegment(); segment_node;
		     segment_node = segments.GetNextSegment(*segment_node)) {
			auto &segment = segment_node->GetNode();
			if (segment.GetCompressionFunction().type != CompressionType::COMPRESSION_DICT_FSST ||
			    segment.segment_type != ColumnSegmentType::PERSISTENT || segment.GetBlockId() == INVALID_BLOCK) {
				return false;
			}
			auto split = dict_fsst::SegmentSplit(segment);
			const auto entry =
			    translations->FindEntry(segment.GetBlockId(), NumericCast<uint32_t>(segment.GetBlockOffset()));
			if (entry == DConstants::INVALID_INDEX || (split && entries[entry].dict_count != split->dict_count) ||
			    entries[entry].rows != segment.count.load()) {
				return false;
			}
			dict_fsst::LinkSegmentTranslation(segment, translations, entry);
			rows += segment.count.load();
			linked++;
		}
	}
	return linked == entries.size();
}

//! Whether a publication in this process links lazily (set once, never cleared)
static atomic<bool> lazy_link_active {false};

bool LazyLinkActive() {
	return lazy_link_active.load();
}

//! Link the segments of one row group's column to their entries in `translations`; a segment they do not hold (a
//! mismatch the table unchanged since its load excludes) is linked to no entry, which a codes-only read of it refuses
static void LinkColumnSegments(ColumnData &column, const shared_ptr<PersistedTranslations> &translations) {
	auto &entries = translations->Entries();
	auto &segments = column.GetSegmentTree();
	for (auto segment_node = segments.GetRootSegment(); segment_node;
	     segment_node = segments.GetNextSegment(*segment_node)) {
		auto &segment = segment_node->GetNode();
		idx_t entry = DConstants::INVALID_INDEX;
		if (segment.GetCompressionFunction().type == CompressionType::COMPRESSION_DICT_FSST &&
		    segment.segment_type == ColumnSegmentType::PERSISTENT && segment.GetBlockId() != INVALID_BLOCK) {
			auto split = dict_fsst::SegmentSplit(segment);
			entry = translations->FindEntry(segment.GetBlockId(), NumericCast<uint32_t>(segment.GetBlockOffset()));
			if (entry != DConstants::INVALID_INDEX &&
			    ((split && entries[entry].dict_count != split->dict_count) ||
			     entries[entry].rows != segment.count.load())) {
				entry = DConstants::INVALID_INDEX;
			}
		}
		dict_fsst::LinkSegmentTranslation(segment, translations, entry);
	}
}

void LinkLoadedColumn(const DataTableInfo &info, idx_t storage_index, ColumnData &column) {
	auto translations = FindPersistedTranslations(info, storage_index);
	if (translations && translations->lazy_link.load()) {
		LinkColumnSegments(column, translations);
	}
}

shared_ptr<ColumnDictionary> PublishPersisted(DataTable &table, idx_t storage_index) {
	if (!PersistedTranslationsEnabled() || !DictGlobalEnabled()) {
		return nullptr;
	}
	optional_ptr<const ColumnDefinition> definition;
	for (auto &column : table.Columns()) {
		if (column.StorageOid() == storage_index) {
			definition = &column;
			break;
		}
	}
	if (!definition || definition->Type().id() != LogicalTypeId::VARCHAR ||
	    !StringType::GetCollation(definition->Type()).empty()) {
		return nullptr;
	}
	auto translations = FindPersistedTranslations(*table.GetDataTableInfo(), storage_index);
	if (!translations) {
		return nullptr;
	}
	auto &collection = *table.GetRowGroupCollection();
	// the translations cover the table's rows only while no row lies outside them (an append not checkpointed) and
	// the column carries no update: otherwise the column is read as without stored translations
	if (translations->column.rows != table.GetTotalRows()) {
		return nullptr;
	}
	auto row_groups = collection.GetRowGroups();
	for (auto node = row_groups->GetRootSegment(); node; node = row_groups->GetNextSegment(*node)) {
		auto &row_group = node->GetNode();
		if (row_group.IsColumnLoaded(storage_index) && row_group.GetRawColumnData(storage_index).HasUpdates()) {
			return nullptr;
		}
	}
	auto registry_entry = TableEntry(table.GetDataTableInfo(), true);
	idx_t epoch;
	{
		lock_guard<mutex> guard(registry_entry->lock);
		epoch = registry_entry->epoch.load();
		auto existing = registry_entry->persisted_columns.find(storage_index);
		if (existing != registry_entry->persisted_columns.end()) {
			auto &dict = existing->second;
			if (dict->epoch == epoch && dict->collection == &collection && dict->persisted == translations) {
				return dict;
			}
		}
	}
	idx_t rows;
	if (collection.TryGetExactLoadedCount(rows) && rows == translations->column.rows && rows == table.GetTotalRows()) {
		// the table unchanged since its load (no append, delete or update): its segments are the ones the translations
		// were built over at the checkpoint that wrote both, so the column is not walked here. Each segment is linked
		// where its row group's column is loaded: now for the loads already done (no read), at the load for the
		// others (RowGroup::LoadColumn, under the row group's lock: a load either sees the flag or is seen below)
		translations->Entries();
		translations->lazy_link = true;
		lazy_link_active = true;
		for (auto node = row_groups->GetRootSegment(); node; node = row_groups->GetNextSegment(*node)) {
			auto &row_group = node->GetNode();
			if (row_group.IsColumnLoadedLocked(storage_index)) {
				LinkColumnSegments(row_group.GetRawColumnData(storage_index), translations);
			}
		}
	} else {
		// a write since the load: the whole column walked and linked, its metadata of every row group requested at
		// once first (each load then finds its blocks read or in flight)
		collection.ReadAheadColumnMetadata({storage_index});
		if (!LinkPersisted(table, storage_index, translations, rows) || rows != table.GetTotalRows() ||
		    rows != translations->column.rows) {
			return nullptr;
		}
	}
	auto entry = make_shared_ptr<ColumnDictionary>(table.db.GetDatabase(), collection, storage_index, epoch,
	                                               definition->Name());
	entry->persisted = translations;
	// the codes-only tag: a child of one invalid slot whose recorded size is the code space; no string is ever read
	// through it (a codes-only plan proves its consumers read codes only)
	entry->child = make_buffer<VectorChildBuffer>(Vector(LogicalType::VARCHAR, 1));
	FlatVector::Validity(entry->child->data).SetInvalid(0);
	entry->child->id = string(CODES_ONLY_ID_PREFIX) + UUID::ToString(UUID::GenerateRandomUUID());
	entry->child->size = translations->Count();
	entry->id = entry->child->id;
	entry->count = translations->Count();
	entry->rows_covered = rows;
	entry->state.store(static_cast<uint8_t>(ColumnState::PUBLISHED), std::memory_order_release);
	{
		auto &registry = CodesOnlyIds();
		lock_guard<mutex> guard(registry.lock);
		registry.ids[entry->id] = translations;
	}
	bool replaced;
	{
		lock_guard<mutex> guard(registry_entry->lock);
		auto &slot = registry_entry->persisted_columns[storage_index];
		if (slot && slot->epoch == epoch && slot->collection == &collection && slot->persisted == translations) {
			return slot;
		}
		replaced = slot != nullptr;
		slot = entry;
	}
	// a first publication changes no plan made before it (Published() is unchanged, and a plan that could type these
	// translations publishes them while it plans), so the version moves only when an entry is replaced: no plan typed on
	// the old entry is then reused (a client's plan cache)
	if (replaced) {
		publication_version.fetch_add(1, std::memory_order_release);
	}
	return entry;
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
	//! the rows of the segments translated (ColumnDictionary::rows_covered)
	idx_t rows_covered = 0;
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
			builder.rows_covered += segment.count;
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
	entry.rows_covered = builder.rows_covered;
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
			builder.rows_covered += state.segments[k].get().count;
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
		if (marked->codes_only) {
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
		if (!typed || typed->emit_codes) {
			// an untyped column, or a key whose codes a projection above the Top-N decodes
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
		if (!overflowed && !dict.persisted) {
			output.data[col].Dictionary(dict.child, sel);
			continue;
		}
		// an overflow code, or a column whose strings are read from the segments where they first occur: a flat
		// string vector
		auto &target = output.data[col];
		target.SetVectorType(VectorType::FLAT_VECTOR);
		auto strings = FlatVector::GetData<string_t>(target);
		auto &validity = FlatVector::Validity(target);
		auto child_strings = dict.persisted ? nullptr : FlatVector::GetData<string_t>(dict.child->data);
		auto &overflow_entry = *overflow[key_idx];
		lock_guard<mutex> guard(overflow_entry.lock);
		for (idx_t i = 0; i < count; i++) {
			auto code = sel.get_index(i);
			if (code == 0) {
				validity.SetInvalid(i);
				continue;
			}
			if (code < dict.count) {
				strings[i] = dict.persisted ? dict.persisted->Fetch(target, UnsafeNumericCast<uint32_t>(code))
				                            : StringVector::AddStringOrBlob(target, child_strings[code]);
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

//===--------------------------------------------------------------------===//
// Codes-only scans
//! Whether `expr` references any of `indexes` through a BOUND_REF
static bool ReferencesAny(const Expression &expr, const unordered_set<idx_t> &indexes) {
	bool found = false;
	ExpressionIterator::VisitExpression<BoundReferenceExpression>(
	    expr, [&](const BoundReferenceExpression &ref) { found = found || indexes.count(ref.index) > 0; });
	return found;
}

//! The scan output position input column `index` of the chain's top resolves to (PROJECTION* / FILTER* BOUND_REFs)
static optional_idx ScanPosition(const vector<reference<PhysicalOperator>> &chain, idx_t index) {
	for (auto &operator_ref : chain) {
		auto &op = operator_ref.get();
		if (op.type == PhysicalOperatorType::FILTER) {
			continue;
		}
		auto &projection = op.Cast<PhysicalProjection>();
		if (index >= projection.select_list.size() ||
		    projection.select_list[index]->GetExpressionType() != ExpressionType::BOUND_REF) {
			return optional_idx();
		}
		index = projection.select_list[index]->Cast<BoundReferenceExpression>().index;
	}
	return index;
}

//! Whether a codes-only scan of the column at input `chunk_index` of the chain's top is sound: no FILTER stands between
//! the scan and the consumer, the column is read by nothing there but BOUND_REFs that carry it to exactly that one input
//! column, its pushed filter (if any) is decided on codes, and the scan emits it once; the consumer reads only its codes
static bool CodesOnlyChain(const vector<reference<PhysicalOperator>> &chain, PhysicalTableScan &scan,
                           idx_t chunk_index) {
	auto position = ScanPosition(chain, chunk_index);
	if (!position.IsValid()) {
		return false;
	}
	const idx_t output_position = position.GetIndex();
	auto column_of = [&](idx_t output) -> optional_idx {
		const idx_t scan_index = scan.projection_ids.empty() ? output : scan.projection_ids[output];
		if (scan_index >= scan.column_ids.size()) {
			return optional_idx();
		}
		return scan_index;
	};
	auto scan_index = column_of(output_position);
	if (!scan_index.IsValid()) {
		return false;
	}
	const idx_t outputs = scan.projection_ids.empty() ? scan.column_ids.size() : scan.projection_ids.size();
	for (idx_t output = 0; output < outputs; output++) {
		auto other = column_of(output);
		if (output != output_position && other.IsValid() &&
		    scan.column_ids[other.GetIndex()] == scan.column_ids[scan_index.GetIndex()]) {
			return false;
		}
	}
	if (scan.table_filters) {
		auto found = scan.table_filters->filters.find(scan_index.GetIndex());
		if (found != scan.table_filters->filters.end() && !CodeTranslatable(*found->second)) {
			return false;
		}
	}
	unordered_set<idx_t> carrying {output_position};
	for (idx_t i = chain.size(); i > 0; i--) {
		auto &op = chain[i - 1].get();
		if (op.type == PhysicalOperatorType::FILTER) {
			// a FILTER caches its small output chunks (a CachingPhysicalOperator), and caching appends a chunk: the
			// append copies the column's strings through its codes, and a codes-only vector holds no strings (its
			// dictionary is a single invalid slot)
			return false;
		}
		auto &projection = op.Cast<PhysicalProjection>();
		unordered_set<idx_t> next;
		for (idx_t out = 0; out < projection.select_list.size(); out++) {
			auto &expr = *projection.select_list[out];
			if (expr.GetExpressionType() == ExpressionType::BOUND_REF) {
				if (carrying.count(expr.Cast<BoundReferenceExpression>().index)) {
					next.insert(out);
				}
			} else if (ReferencesAny(expr, carrying)) {
				return false;
			}
		}
		carrying = std::move(next);
	}
	return carrying.size() == 1 && carrying.count(chunk_index) == 1;
}

//! The chain PROJECTION* / FILTER* over one seq_scan of a DuckDB table, or false
static bool ScanChain(PhysicalOperator &child, vector<reference<PhysicalOperator>> &chain,
                      optional_ptr<PhysicalTableScan> &scan) {
	reference<PhysicalOperator> current(child);
	while (current.get().type == PhysicalOperatorType::PROJECTION ||
	       current.get().type == PhysicalOperatorType::FILTER) {
		if (current.get().children.size() != 1) {
			return false;
		}
		chain.push_back(current);
		current = current.get().children[0];
	}
	if (current.get().type != PhysicalOperatorType::TABLE_SCAN) {
		return false;
	}
	scan = &current.get().Cast<PhysicalTableScan>();
	return scan->function.name == "seq_scan" && scan->bind_data;
}

//! The scan reads the column codes only, through the dictionary's stored translations (the column must be marked)
static void SetCodesOnly(PhysicalTableScan &scan, const shared_ptr<ColumnDictionary> &dict,
                         shared_ptr<atomic<bool>> admitted = nullptr) {
	auto publication = ScanPublicationOf(scan.bind_data.get());
	if (!publication || !dict->persisted) {
		return;
	}
	for (auto &column : publication->columns) {
		if (column.storage_index == dict->storage_index) {
			column.codes_only = dict->persisted;
			column.codes_only_dict = dict;
			column.codes_only_admitted = admitted;
		}
	}
}

//===--------------------------------------------------------------------===//
// Plans over stale stored translations
static bool ScanTranslationsCurrent(ClientContext &context, const PhysicalOperator &op) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		auto publication = scan.bind_data ? ScanPublicationOf(scan.bind_data.get()) : nullptr;
		if (publication) {
			for (auto &column : publication->columns) {
				if (!column.codes_only_dict) {
					continue;
				}
				auto &table = scan.bind_data->Cast<TableScanBindData>().table;
				if (!table.IsDuckTable()) {
					return false;
				}
				auto &storage = table.Cast<DuckTableEntry>().GetStorage();
				if (column.codes_only_admitted) {
					// the consumer decides at its execution whether the translations cover the state it reads (refused,
					// the scan reads strings); the plan re-binds only once a checkpoint replaced the translations
					if (FindPersistedTranslations(*storage.GetDataTableInfo(), column.storage_index) != column.codes_only) {
						return false;
					}
					continue;
				}
				auto current = PublishPersisted(storage, column.storage_index);
				if (current != column.codes_only_dict) {
					return false;
				}
				// a column read codes-only at every execution: the rows of this transaction's local storage hold strings
				// (an append outside the translations or an update makes PublishPersisted's entry null above)
				if (LocalStorage::Get(context, storage.db).Find(storage)) {
					return false;
				}
			}
		}
	}
	for (auto &child : op.GetChildren()) {
		if (!ScanTranslationsCurrent(context, child.get())) {
			return false;
		}
	}
	return true;
}

bool PlanTranslationsCurrent(ClientContext &context, const PhysicalOperator &root) {
	return ScanTranslationsCurrent(context, root);
}

void ThrowStaleTranslations(const string &column_name) {
	throw InvalidInputException("The stored translations of column \"%s\" changed after this statement was planned (a "
	                            "checkpoint since); prepare or run the statement again",
	                            column_name);
}

namespace {
//! Re-binds a prepared statement whose plan reads stale stored translations (prepared before a checkpoint)
class StoredTranslationPlans : public ClientContextState {
public:
	static bool Stale(ClientContext &context, PreparedStatementData &prepared) {
		return prepared.physical_plan && !PlanTranslationsCurrent(context, prepared.physical_plan->Root());
	}
	RebindQueryInfo OnExecutePrepared(ClientContext &context, PreparedStatementCallbackInfo &info,
	                                  RebindQueryInfo current_rebind) override {
		if (current_rebind == RebindQueryInfo::ATTEMPT_TO_REBIND) {
			return current_rebind;
		}
		return Stale(context, info.prepared_statement) ? RebindQueryInfo::ATTEMPT_TO_REBIND : RebindQueryInfo::DO_NOT_REBIND;
	}
	RebindQueryInfo OnRebindPreparedStatement(ClientContext &context, BindPreparedStatementCallbackInfo &info,
	                                          RebindQueryInfo current_rebind) override {
		if (current_rebind == RebindQueryInfo::ATTEMPT_TO_REBIND) {
			return current_rebind;
		}
		return Stale(context, info.prepared_statement) ? RebindQueryInfo::ATTEMPT_TO_REBIND : RebindQueryInfo::DO_NOT_REBIND;
	}
};
} // namespace

void NoteStoredTranslationPlan(ClientContext &context) {
	context.registered_state->GetOrCreate<StoredTranslationPlans>("dict_global_stored_translation_plans");
}

void MarkCodesOnly(PhysicalOperator &child, idx_t chunk_index, const shared_ptr<ColumnDictionary> &dict,
                   shared_ptr<atomic<bool>> admitted) {
	vector<reference<PhysicalOperator>> chain;
	optional_ptr<PhysicalTableScan> scan;
	if (!dict || !dict->persisted || !ScanChain(child, chain, scan) || !CodesOnlyChain(chain, *scan, chunk_index)) {
		return;
	}
	SetCodesOnly(*scan, dict, std::move(admitted));
}

//===--------------------------------------------------------------------===//
// Late decode above a Top-N
namespace {
struct LateDecodeData : public FunctionData {
	LateDecodeData(shared_ptr<CodeKeys> keys_p, idx_t key_index_p) : keys(std::move(keys_p)), key_index(key_index_p) {
	}
	shared_ptr<CodeKeys> keys;
	idx_t key_index;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<LateDecodeData>(keys, key_index);
	}
	bool Equals(const FunctionData &other) const override {
		auto &cast = other.Cast<LateDecodeData>();
		return keys == cast.keys && key_index == cast.key_index;
	}
};

void LateDecodeFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &data = expr.bind_info->Cast<LateDecodeData>();
	auto &key = data.keys->keys[data.key_index];
	auto &dict = *key.dict;
	const idx_t count = args.size();
	UnifiedVectorFormat format;
	args.data[0].ToUnifiedFormat(count, format);
	auto codes = UnifiedVectorFormat::GetData<int32_t>(format);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto strings = FlatVector::GetData<string_t>(result);
	auto &validity = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		const auto idx = format.sel->get_index(i);
		if (!format.validity.RowIsValid(idx) || codes[idx] == 0) {
			validity.SetInvalid(i);
			continue;
		}
		const auto code = UnsafeNumericCast<idx_t>(codes[idx]);
		if (code < dict.count) {
			strings[i] = dict.persisted->Fetch(result, UnsafeNumericCast<uint32_t>(code));
		} else {
			auto &overflow_entry = *data.keys->overflow[data.key_index];
			lock_guard<mutex> guard(overflow_entry.lock);
			strings[i] = StringVector::AddStringOrBlob(result, overflow_entry.strings[code - dict.count]);
		}
	}
}
} // namespace

vector<LateDecode> PlanLateDecode(PhysicalOperator &child, const vector<BoundOrderByNode> &orders) {
	vector<LateDecode> result;
	if (!PersistedTranslationsEnabled()) {
		return result;
	}
	// the chain: PROJECTION* over a grouped hash aggregate with code keys
	vector<reference<PhysicalOperator>> projections;
	reference<PhysicalOperator> current(child);
	while (current.get().type == PhysicalOperatorType::PROJECTION) {
		if (current.get().children.size() != 1) {
			return result;
		}
		projections.push_back(current);
		current = current.get().children[0];
	}
	if (current.get().type != PhysicalOperatorType::HASH_GROUP_BY) {
		return result;
	}
	auto &aggregate = current.get();
	auto keys = FindCodeKeys(&aggregate);
	if (!keys) {
		return result;
	}
	struct Carried {
		idx_t key_index;
		//! the positions carrying the key at each level, from the aggregate's output up to the Top-N's input
		vector<vector<idx_t>> positions;
	};
	vector<Carried> carried;
	for (idx_t k = 0; k < keys->keys.size(); k++) {
		auto &key = keys->keys[k];
		if (!key.dict || !key.dict->persisted || key.emit_codes) {
			continue;
		}
		Carried entry;
		entry.key_index = k;
		unordered_set<idx_t> set {key.group_index};
		entry.positions.push_back({key.group_index});
		bool sound = true;
		for (idx_t i = projections.size(); i > 0 && sound; i--) {
			auto &projection = projections[i - 1].get().Cast<PhysicalProjection>();
			unordered_set<idx_t> next;
			vector<idx_t> next_positions;
			for (idx_t out = 0; out < projection.select_list.size(); out++) {
				auto &expr = *projection.select_list[out];
				if (expr.GetExpressionType() == ExpressionType::BOUND_REF) {
					if (set.count(expr.Cast<BoundReferenceExpression>().index)) {
						next.insert(out);
						next_positions.push_back(out);
					}
				} else if (ReferencesAny(expr, set)) {
					sound = false;
				}
			}
			set = std::move(next);
			entry.positions.push_back(std::move(next_positions));
		}
		for (auto &order : orders) {
			sound = sound && !ReferencesAny(*order.expression, set);
		}
		if (sound && !set.empty()) {
			carried.push_back(std::move(entry));
		}
	}
	for (auto &entry : carried) {
		auto &key = keys->keys[entry.key_index];
		key.emit_codes = true;
		aggregate.types[key.group_index] = LogicalType::INTEGER;
		for (idx_t level = 1; level < entry.positions.size(); level++) {
			auto &projection = projections[projections.size() - level].get().Cast<PhysicalProjection>();
			for (auto out : entry.positions[level]) {
				projection.select_list[out]->return_type = LogicalType::INTEGER;
				projection.types[out] = LogicalType::INTEGER;
			}
		}
		for (auto position : entry.positions.back()) {
			result.push_back(LateDecode {position, keys, entry.key_index});
		}
	}
	return result;
}

unique_ptr<Expression> LateDecodeExpression(const LateDecode &decode, idx_t index, const LogicalType &type) {
	ScalarFunction function("__dict_global_decode", {LogicalType::INTEGER}, type, LateDecodeFunction);
	vector<unique_ptr<Expression>> children;
	children.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, index));
	return make_uniq<BoundFunctionExpression>(type, std::move(function), std::move(children),
	                                          make_uniq<LateDecodeData>(decode.keys, decode.key_index));
}

unique_ptr<Expression> CodeDecodeExpression(const shared_ptr<ColumnDictionary> &dict, unique_ptr<Expression> child,
                                            const LogicalType &type) {
	// one emitted key over the dictionary's stored translations; its overflow is never taken (every code a stored one)
	auto keys = make_shared_ptr<CodeKeys>();
	CodeKey key;
	key.group_index = 0;
	key.chunk_index = 0;
	key.dict = dict;
	key.emit_codes = true;
	keys->keys.push_back(std::move(key));
	keys->overflow.push_back(make_uniq<CodeKeys::Overflow>());
	keys->column_names = dict->column_name;
	ScalarFunction function("__dict_global_decode", {LogicalType::INTEGER}, type, LateDecodeFunction);
	function.SetFallible();
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(child));
	return make_uniq<BoundFunctionExpression>(type, std::move(function), std::move(children),
	                                          make_uniq<LateDecodeData>(std::move(keys), 0));
}

namespace {
//! __dict_global_codes: a codes-only vector's codes (its dictionary selection) as INTEGER, NULL for code 0. Fail
//! closed: any other vector (strings, or a dictionary vector that is not over stored translations) throws, never a
//! lookup
void CodesFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &input = args.data[0];
	const idx_t count = args.size();
	if (input.GetVectorType() == VectorType::CONSTANT_VECTOR && ConstantVector::IsNull(input)) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		ConstantVector::SetNull(result, true);
		return;
	}
	if (input.GetVectorType() != VectorType::DICTIONARY_VECTOR ||
	    !CodesOnlyTranslationsOf(DictionaryVector::DictionaryId(input))) {
		throw InternalException("a codes-only vector was expected");
	}
	auto &codes = DictionaryVector::SelVector(input);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<int32_t>(result);
	auto &validity = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		const auto code = codes.get_index(i);
		out[i] = UnsafeNumericCast<int32_t>(code);
		if (code == 0) {
			validity.SetInvalid(i);
		}
	}
}
} // namespace

unique_ptr<Expression> CodesExpression(unique_ptr<Expression> child) {
	ScalarFunction function("__dict_global_codes", {LogicalType::VARCHAR}, LogicalType::INTEGER, CodesFunction);
	// it throws on a vector that is not codes-only: never evaluated over a dictionary's entries instead of its rows
	function.SetFallible();
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(child));
	return make_uniq<BoundFunctionExpression>(LogicalType::INTEGER, std::move(function), std::move(children), nullptr);
}

namespace {
struct ByteLengthData : public FunctionData {
	ByteLengthData(shared_ptr<ColumnDictionary> dict_p, bool bits_p) : dict(std::move(dict_p)), bits(bits_p) {
	}
	//! the column's publication entry over its stored translations, which store the byte lengths
	shared_ptr<ColumnDictionary> dict;
	//! bit_length: the byte length times 8
	bool bits;

	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ByteLengthData>(dict, bits);
	}
	bool Equals(const FunctionData &other) const override {
		auto &cast = other.Cast<ByteLengthData>();
		return dict == cast.dict && bits == cast.bits;
	}
};

//! __dict_global_byte_length: strlen (or bit_length) of a codes-only vector, each row's code looked up in the stored byte
//! lengths, NULL for code 0. Fail closed: any vector that is not a codes-only vector of the bind data's translations
//! (strings, or codes of other translations) throws, never a lookup
void ByteLengthFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &expr = state.expr.Cast<BoundFunctionExpression>();
	auto &data = expr.bind_info->Cast<ByteLengthData>();
	auto &input = args.data[0];
	const idx_t count = args.size();
	auto &translations = *data.dict->persisted;
	// a vector over the entry's own tag child is codes-only over these translations; any other is resolved by its id
	if (input.GetVectorType() != VectorType::DICTIONARY_VECTOR ||
	    (&DictionaryVector::Child(input) != &data.dict->child->data &&
	     CodesOnlyTranslationsOf(DictionaryVector::DictionaryId(input)) != data.dict->persisted)) {
		throw InternalException("a codes-only vector of the stored byte lengths' translations was expected");
	}
	auto lengths = translations.Lengths();
	if (!lengths) {
		throw InternalException("Stored column translations without byte lengths read for them");
	}
	const idx_t code_count = translations.Count();
	const int64_t factor = data.bits ? 8 : 1;
	auto &codes = DictionaryVector::SelVector(input);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<int64_t>(result);
	auto &validity = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		const auto code = codes.get_index(i);
		if (code == 0) {
			validity.SetInvalid(i);
			continue;
		}
		if (code >= code_count) {
			throw IOException("Stored column translations: a code outside the byte-length table - the database file "
			                  "appears corrupted");
		}
		out[i] = int64_t(lengths[code]) * factor;
	}
}

unique_ptr<Expression> ByteLengthExpression(const shared_ptr<ColumnDictionary> &dict, bool bits,
                                            unique_ptr<Expression> child) {
	ScalarFunction function("__dict_global_byte_length", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                        ByteLengthFunction);
	// it throws on a vector that is not codes-only: never evaluated over a dictionary's entries instead of its rows
	function.SetFallible();
	vector<unique_ptr<Expression>> children;
	children.push_back(std::move(child));
	return make_uniq<BoundFunctionExpression>(LogicalType::BIGINT, std::move(function), std::move(children),
	                                          make_uniq<ByteLengthData>(dict, bits));
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
		if (!dict && StringType::GetCollation(resolved.type).empty() &&
		    StringType::GetCollation(group.return_type).empty() && grouping_set_count <= 1 &&
		    std::find(read_indexes.begin(), read_indexes.end(), chunk_index) == read_indexes.end() &&
		    ScanPublicationOf(scan.bind_data.get()) && ScanPublicationOf(scan.bind_data.get())->Find(resolved.storage_index) &&
		    !LocalStorage::Get(context, duck_table.GetStorage().db).Find(duck_table.GetStorage()) &&
		    scan.estimated_cardinality >= idx_t(kStoredCodeKeysMinScanRows) &&
		    CodesOnlyChain(chain, scan, chunk_index)) {
			dict = PublishPersisted(duck_table.GetStorage(), resolved.storage_index);
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
		if (key.dict->persisted) {
			result->column_names += " (stored)";
		}
		result->keys.push_back(std::move(key));
	}
	if (result->keys.empty()) {
		return nullptr;
	}
	// the wide guard: with a packed group width (plan_aggregate's key_bytes) above the fused kernel's key bytes
	// (WideGuardKeyBytes) once these keys are typed, the code key would land on the general hash table, where it is
	// slower than the string key; such keys stay strings (the bound is the fused kernel's key width)
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
		if (key_bytes > WideGuardKeyBytes()) {
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
		if (key.dict->persisted) {
			// the scan emits this key's codes only; a prepared plan re-binds once these translations go stale
			SetCodesOnly(scan, key.dict);
			NoteStoredTranslationPlan(context);
		}
	}
	return result;
}

shared_ptr<ColumnDictionary> PlanPublishedColumn(PhysicalOperator &child, idx_t chunk_index, const LogicalType &type,
                                                 optional_ptr<DataTable> &table) {
	if (type.InternalType() != PhysicalType::VARCHAR || !StringType::GetCollation(type).empty()) {
		return nullptr;
	}
	// the chain: PROJECTION* / FILTER* over one seq_scan of a DuckDB table (PlanCodeKeys's chain)
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
	auto resolved = ResolveColumn(chain, scan, duck_table, chunk_index);
	if (!resolved.found || resolved.type.InternalType() != PhysicalType::VARCHAR ||
	    !StringType::GetCollation(resolved.type).empty()) {
		return nullptr;
	}
	auto publication = ScanPublicationOf(scan.bind_data.get());
	if (!publication || !publication->Find(resolved.storage_index)) {
		return nullptr;
	}
	shared_ptr<ColumnDictionary> dict;
	if (publication->Publishes(resolved.storage_index)) {
		dict = Published(duck_table.GetStorage(), resolved.storage_index);
	}
	if (!dict && CodesOnlyChain(chain, scan, chunk_index)) {
		dict = PublishPersisted(duck_table.GetStorage(), resolved.storage_index);
	}
	if (dict) {
		table = duck_table.GetStorage();
	}
	return dict;
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
	const bool string_predicate = S1GuardEnabled() && ScanHasStringExpressionFilter(scan, duck_table);
	if (string_predicate && !kStringPredicateStoredCodeKeys) {
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
		// under the string-predicate build guard only a column its consumer can read codes only through stored
		// translations is marked (PlanCodeKeys's persisted conditions), and gated below: never built or emitted over a
		// published dictionary; every other column stays unmarked as under the guard
		if (string_predicate &&
		    (scan.estimated_cardinality < idx_t(kStoredCodeKeysMinScanRows) ||
		     LocalStorage::Get(context, duck_table.GetStorage().db).Find(duck_table.GetStorage()) ||
		     !CodesOnlyChain(chain, scan, index) || !PublishPersisted(duck_table.GetStorage(), resolved.storage_index))) {
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
		column.gated = string_predicate ||
		               static_cast<double>(column.estimated_rows) < GATE_SHARE * static_cast<double>(column.table_rows);
		publication.columns.push_back(std::move(column));
	}
}

void MarkFilterOnlyCodes(ClientContext &context, PhysicalTableScan &scan) {
	if (!kFilterOnlyCodesOnly || !PersistedTranslationsEnabled() || !DictGlobalEnabled() || !scan.table_filters ||
	    scan.projection_ids.empty() || scan.function.name != "seq_scan" || !scan.bind_data ||
	    scan.estimated_cardinality < idx_t(kStoredCodeKeysMinScanRows)) {
		return;
	}
	auto &table_entry = scan.bind_data->Cast<TableScanBindData>().table;
	if (!table_entry.IsDuckTable()) {
		return;
	}
	auto &duck_table = table_entry.Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	// this transaction's local rows hold strings (PlanCodeKeys's refusal; a prepared plan re-binds once they appear)
	if (LocalStorage::Get(context, storage.db).Find(storage)) {
		return;
	}
	for (auto &entry : scan.table_filters->filters) {
		const idx_t scan_index = entry.first;
		if (scan_index >= scan.column_ids.size() || !CodeTranslatable(*entry.second) ||
		    std::find(scan.projection_ids.begin(), scan.projection_ids.end(), scan_index) != scan.projection_ids.end()) {
			continue;
		}
		auto &column_index = scan.column_ids[scan_index];
		if (column_index.IsRowIdColumn() || column_index.IsVirtualColumn() || column_index.HasChildren() ||
		    !column_index.HasPrimaryIndex()) {
			continue;
		}
		bool once = true;
		for (idx_t i = 0; i < scan.column_ids.size(); i++) {
			once = once && (i == scan_index || !(scan.column_ids[i] == column_index));
		}
		auto &column = duck_table.GetColumns().GetColumn(LogicalIndex(column_index.GetPrimaryIndex()));
		if (!once || column.Generated() || column.Type().id() != LogicalTypeId::VARCHAR ||
		    !StringType::GetCollation(column.Type()).empty()) {
			continue;
		}
		auto dict = PublishPersisted(storage, column.StorageOid());
		if (!dict) {
			continue;
		}
		auto publishing = dynamic_cast<PublishingTableScanBindData *>(scan.bind_data.get());
		if (!publishing) {
			auto replacement = make_uniq<PublishingTableScanBindData>(scan.bind_data->Cast<TableScanBindData>());
			publishing = replacement.get();
			scan.bind_data = std::move(replacement);
		}
		if (!publishing->publication) {
			publishing->publication = make_shared_ptr<ScanPublication>();
		}
		if (publishing->publication->Find(column.StorageOid())) {
			continue;
		}
		// marked but gated: the column is never built or emitted over a published dictionary, only read codes only
		ScanPublication::Column marked;
		marked.storage_index = column.StorageOid();
		marked.name = column.Name();
		marked.gated = true;
		marked.filter_only = true;
		publishing->publication->columns.push_back(std::move(marked));
		SetCodesOnly(scan, dict);
		NoteStoredTranslationPlan(context);
	}
}

//! Whether `expr` is strlen or bit_length of VARCHAR over a bare reference to one of `positions`
static bool IsByteLengthConsumer(const Expression &expr, const unordered_set<idx_t> &positions) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	if ((function.function.name != "strlen" && function.function.name != "bit_length") ||
	    function.children.size() != 1 || function.function.arguments.size() != 1 ||
	    function.function.arguments[0].id() != LogicalTypeId::VARCHAR ||
	    function.return_type.id() != LogicalTypeId::BIGINT) {
		return false;
	}
	auto &child = *function.children[0];
	return child.GetExpressionType() == ExpressionType::BOUND_REF &&
	       positions.count(child.Cast<BoundReferenceExpression>().index) > 0;
}

void MarkByteLengthConsumers(ClientContext &context, PhysicalOperator &plan) {
	if (!kDictionaryEntryLengths || !PersistedTranslationsEnabled() || !DictGlobalEnabled() ||
	    plan.type != PhysicalOperatorType::PROJECTION || plan.children.size() != 1) {
		return;
	}
	// off under query verification: the plan's serializer round trip cannot rebuild the internal function
	auto &config = ClientConfig::GetConfig(context);
	if (config.query_verification_enabled || config.verify_serializer) {
		return;
	}
	auto &projection = plan.Cast<PhysicalProjection>();
	// the chain: PROJECTION* over one seq_scan of a DuckDB table, no FILTER (a FILTER's cache appends the codes-only
	// vector, whose dictionary holds no strings)
	vector<reference<PhysicalOperator>> chain;
	optional_ptr<PhysicalTableScan> scan;
	if (!ScanChain(projection.children[0], chain, scan) || scan->dynamic_filters ||
	    scan->estimated_cardinality < idx_t(kStoredCodeKeysMinScanRows)) {
		return;
	}
	for (auto &op : chain) {
		if (op.get().type == PhysicalOperatorType::FILTER) {
			return;
		}
	}
	auto &table_entry = scan->bind_data->Cast<TableScanBindData>().table;
	if (!table_entry.IsDuckTable()) {
		return;
	}
	auto &duck_table = table_entry.Cast<DuckTableEntry>();
	auto &storage = duck_table.GetStorage();
	// this transaction's local rows hold strings (a prepared plan re-binds once they appear)
	if (LocalStorage::Get(context, storage.db).Find(storage)) {
		return;
	}
	const idx_t outputs = scan->projection_ids.empty() ? scan->column_ids.size() : scan->projection_ids.size();
	for (idx_t output = 0; output < outputs; output++) {
		const idx_t scan_index = scan->projection_ids.empty() ? output : scan->projection_ids[output];
		if (scan_index >= scan->column_ids.size()) {
			continue;
		}
		auto &column_index = scan->column_ids[scan_index];
		if (column_index.IsRowIdColumn() || column_index.IsVirtualColumn() || column_index.HasChildren() ||
		    !column_index.HasPrimaryIndex()) {
			continue;
		}
		auto &column = duck_table.GetColumns().GetColumn(LogicalIndex(column_index.GetPrimaryIndex()));
		if (column.Generated() || column.Type().id() != LogicalTypeId::VARCHAR ||
		    !StringType::GetCollation(column.Type()).empty()) {
			continue;
		}
		// the scan emits the column once, and its pushed filter, if any, is decided on codes
		bool once = true;
		for (idx_t other = 0; other < outputs; other++) {
			const idx_t other_index = scan->projection_ids.empty() ? other : scan->projection_ids[other];
			once = once && (other == output || other_index >= scan->column_ids.size() ||
			                !(scan->column_ids[other_index] == column_index));
		}
		if (!once) {
			continue;
		}
		if (scan->table_filters) {
			auto found = scan->table_filters->filters.find(scan_index);
			if (found != scan->table_filters->filters.end() && !CodeTranslatable(*found->second)) {
				continue;
			}
		}
		// the chain carries the column by bare references only (CodesOnlyChain's walk, to any set of positions)
		unordered_set<idx_t> carrying {output};
		bool sound = true;
		for (idx_t i = chain.size(); i > 0 && sound; i--) {
			auto &lower = chain[i - 1].get().Cast<PhysicalProjection>();
			unordered_set<idx_t> next;
			for (idx_t out = 0; out < lower.select_list.size(); out++) {
				auto &expr = *lower.select_list[out];
				if (expr.GetExpressionType() == ExpressionType::BOUND_REF) {
					if (carrying.count(expr.Cast<BoundReferenceExpression>().index)) {
						next.insert(out);
					}
				} else if (ReferencesAny(expr, carrying)) {
					sound = false;
				}
			}
			carrying = std::move(next);
		}
		if (!sound || carrying.empty()) {
			continue;
		}
		// the projection reads the column only as the sole argument of strlen or bit_length, and drops it
		vector<idx_t> consumers;
		for (idx_t out = 0; out < projection.select_list.size() && sound; out++) {
			auto &expr = *projection.select_list[out];
			if (IsByteLengthConsumer(expr, carrying)) {
				consumers.push_back(out);
			} else if (ReferencesAny(expr, carrying)) {
				sound = false;
			}
		}
		if (!sound || consumers.empty()) {
			continue;
		}
		auto dict = PublishPersisted(storage, column.StorageOid());
		if (!dict || !dict->persisted || !dict->persisted->HasLengths()) {
			continue;
		}
		auto publishing = dynamic_cast<PublishingTableScanBindData *>(scan->bind_data.get());
		if (!publishing) {
			auto replacement = make_uniq<PublishingTableScanBindData>(scan->bind_data->Cast<TableScanBindData>());
			publishing = replacement.get();
			scan->bind_data = std::move(replacement);
		}
		if (!publishing->publication) {
			publishing->publication = make_shared_ptr<ScanPublication>();
		}
		if (publishing->publication->Find(column.StorageOid())) {
			continue;
		}
		// marked but gated: never built or emitted over a published dictionary, only read codes only
		ScanPublication::Column marked;
		marked.storage_index = column.StorageOid();
		marked.name = column.Name();
		marked.gated = true;
		marked.byte_length = true;
		publishing->publication->columns.push_back(std::move(marked));
		SetCodesOnly(*scan, dict);
		for (auto out : consumers) {
			auto &function = projection.select_list[out]->Cast<BoundFunctionExpression>();
			const bool bits = function.function.name == "bit_length";
			projection.select_list[out] = ByteLengthExpression(dict, bits, std::move(function.children[0]));
		}
		NoteStoredTranslationPlan(context);
	}
}

bool MarkCodeGroupKey(ClientContext &context, LogicalGet &get, idx_t column_index,
                      const shared_ptr<ColumnDictionary> &dict) {
	// a get already carrying dynamic filters is refused: a runtime filter on a codes-only column must be decided on
	// codes
	if (get.dynamic_filters || !dict || !dict->persisted || get.function.name != "seq_scan" || !get.bind_data) {
		return false;
	}
	auto table = get.GetTable();
	auto &column_ids = get.GetColumnIds();
	if (!table || !table->IsDuckTable() || column_index >= column_ids.size()) {
		return false;
	}
	auto &column_id = column_ids[column_index];
	if (column_id.IsRowIdColumn() || column_id.IsVirtualColumn() || column_id.HasChildren() ||
	    !column_id.HasPrimaryIndex()) {
		return false;
	}
	// the publication marks a storage column for the whole scan: the scan must read the column once
	for (idx_t i = 0; i < column_ids.size(); i++) {
		if (i != column_index && column_ids[i] == column_id) {
			return false;
		}
	}
	auto &column = table->GetColumns().GetColumn(LogicalIndex(column_id.GetPrimaryIndex()));
	if (column.StorageOid() != dict->storage_index) {
		return false;
	}
	auto publishing = dynamic_cast<PublishingTableScanBindData *>(get.bind_data.get());
	if (!publishing) {
		auto replacement = make_uniq<PublishingTableScanBindData>(get.bind_data->Cast<TableScanBindData>());
		publishing = replacement.get();
		get.bind_data = std::move(replacement);
	}
	if (!publishing->publication) {
		publishing->publication = make_shared_ptr<ScanPublication>();
	}
	auto existing = publishing->publication->Find(dict->storage_index);
	if (existing) {
		// a second code group over the same column reads the same codes
		return existing->group_key_codes && existing->codes_only_dict == dict;
	}
	// marked but gated: never built or emitted over a published dictionary, only read codes only
	ScanPublication::Column marked;
	marked.storage_index = dict->storage_index;
	marked.name = column.Name();
	marked.gated = true;
	marked.group_key_codes = true;
	marked.codes_only = dict->persisted;
	marked.codes_only_dict = dict;
	publishing->publication->columns.push_back(std::move(marked));
	NoteStoredTranslationPlan(context);
	return true;
}

string CodesOnlyColumnNames(const FunctionData *bind_data) {
	string result;
	auto publication = ScanPublicationOf(bind_data);
	if (!publication) {
		return result;
	}
	for (auto &column : publication->columns) {
		if (column.codes_only) {
			result += (result.empty() ? "" : "\n") + column.name +
			          (column.filter_only       ? " (filter only)"
			           : column.group_key_codes ? " (group key)"
			           : column.byte_length     ? " (byte length)"
			                                    : "");
		}
	}
	return result;
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
