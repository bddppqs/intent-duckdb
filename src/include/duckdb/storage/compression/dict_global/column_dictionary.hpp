//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/compression/dict_global/column_dictionary.hpp
//
// An in-memory, read-only global dictionary per admitted VARCHAR column,
// built complete from the per-segment DICT_FSST dictionaries at the table-scan
// initialisation (in parallel across the worker threads, within a memory budget)
// and published once under a per-build id, and the code keys a physical hash
// aggregate may type from it.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/storage/object_cache.hpp"

namespace duckdb {
class ClientContext;
class ColumnSegment;
class DataTable;
class PhysicalOperator;
class PhysicalTableScan;
class Expression;
class LogicalGet;
class RowGroupCollection;
struct BoundOrderByNode;
struct DataTableInfo;
struct FunctionData;

namespace dict_global {
class PersistedTranslations;

//! The memory_limit below which the memory-for-speed mechanisms take their small-memory form (no borrowed group keys,
//! no MIN/MAX string arena, the global-dictionary budget a quarter of the pool); at or above it they keep their
//! full form
static constexpr idx_t SMALL_MEMORY_LIMIT = idx_t(8) << 30;

//===--------------------------------------------------------------------===//
// Compile-time defaults (duckdb/common/tuning_defaults.hpp)
//===--------------------------------------------------------------------===//
//! The global dictionary: the build, the published column vectors, the borrow
bool DictGlobalEnabled();
//! A published admitted group key may be typed as its 4-byte code (with the global dictionary enabled)
bool CodeKeysEnabled();
//! The fused kernel's DISTINCT class attaches over an operator with a code key; off, the operator takes the general
//! aggregate with its keys still typed
bool FusedDistinctEnabled();
//! Each segment's translation carries its own handle on the column's child, which the vectors of that segment's scans
//! reference; off, they reference the column's child itself
bool BorrowChildEnabled();

//===--------------------------------------------------------------------===//
// The column dictionary
//===--------------------------------------------------------------------===//
enum class ColumnState : uint8_t { UNBUILT = 0, BUILDING = 1, PUBLISHED = 2, REFUSED = 3 };

//! One admitted column's global dictionary for one storage epoch; immutable once published
class ColumnDictionary {
public:
	ColumnDictionary(DatabaseInstance &db, const RowGroupCollection &collection, idx_t storage_index, idx_t epoch,
	                 string column_name);
	~ColumnDictionary();

	DatabaseInstance &db;
	//! the row groups it was built from: a storage index is a column's identity only within one collection (an ALTER
	//! builds a new collection and DROP COLUMN renumbers; the entries are erased in every alter constructor, so a
	//! later collection at a reused address never finds an entry of a dead one)
	const RowGroupCollection *const collection;
	const idx_t storage_index;
	const idx_t epoch;
	const string column_name;
	atomic<uint8_t> state;

	//! Published fields (written once before the release store of PUBLISHED)
	buffer_ptr<VectorChildBuffer> child;
	string id;
	idx_t count = 0;
	idx_t bytes = 0;
	//! The string -> code index kept after the publish for the per-chunk lookup of a vector that is not over the child
	unsafe_unique_array<uint32_t> index;
	idx_t index_mask = 0;
	//! Bytes reserved through BufferManager::ReserveMemory (EXTENSION), released with the entry
	idx_t reserved = 0;
	//! The rows of the segments whose dictionaries the build merged (the segments with a translation); a column
	//! whose every committed row lies in them reads rows_covered == DataTable::GetTotalRows()
	idx_t rows_covered = 0;
	//! A column read through stored translations: a codes-only child; a code's string is read where it first occurs
	shared_ptr<PersistedTranslations> persisted;

	ColumnState GetState() const {
		return static_cast<ColumnState>(state.load(std::memory_order_acquire));
	}
	//! The code of a string (a published dictionary): its code, or INVALID_CODE when absent
	uint32_t Lookup(const string_t &value, hash_t hash) const;

	static constexpr uint32_t INVALID_CODE = NumericLimits<uint32_t>::Maximum();
};

//! A handle on a published child (null when BorrowChildEnabled() is false): a control block of its own that owns one
//! reference to `child`, aliased to it, so a copy of the handle counts on that block and the child lives while any copy
//! does
buffer_ptr<VectorChildBuffer> ChildHandle(const buffer_ptr<VectorChildBuffer> &child);

//! One segment's translation, local code -> global code, registered non-evictable in the object cache
class SegmentTranslation : public ObjectCacheEntry {
public:
	SegmentTranslation(shared_ptr<ColumnDictionary> dict_p, unsafe_unique_array<uint32_t> codes_p, idx_t count_p)
	    : dict(std::move(dict_p)), codes(std::move(codes_p)), count(count_p), child(ChildHandle(dict->child)) {
	}
	static string ObjectType() {
		return "global_segment_translation";
	}
	string GetObjectType() override {
		return ObjectType();
	}
	optional_idx GetEstimatedCacheMemory() const override {
		return optional_idx();
	}

	shared_ptr<ColumnDictionary> dict;
	unsafe_unique_array<uint32_t> codes;
	idx_t count;
	//! a codes-only read of a column its scan reads for the filter only: its segment filter decides on local codes and
	//! emits no values
	bool filter_only = false;
	//! This segment's handle on dict->child (ChildHandle; null when kSegmentTranslationChildHandle is off)
	buffer_ptr<VectorChildBuffer> child;

	//! The child a vector of this segment references: the segment's handle, else the column's child
	const buffer_ptr<VectorChildBuffer> &VectorChild() const {
		return child ? child : dict->child;
	}
};

//! The key of a segment's translation entry
string TranslationKey(const string &segment_key);
//! The published translation of a DICT_FSST segment, or null (cheap when nothing is published)
shared_ptr<SegmentTranslation> FindTranslation(ColumnSegment &segment);
//! Release a segment's translation (the segment dies or its key rotates)
void ReleaseTranslation(DatabaseInstance &db, const string &segment_key);

//! The per-table registry: a process-wide map keyed by the table's DataTableInfo, never a member of it (a prebuilt
//! extension carries its own DuckDB built from upstream headers and reads DataTableInfo at the upstream offsets)
struct TableDictionaries {
	explicit TableDictionaries(const shared_ptr<DataTableInfo> &info);
	//! the owner: an entry whose owner expired belongs to a dropped table (its address may be reused)
	weak_ptr<DataTableInfo> owner;
	mutex lock;
	//! the storage epoch: a process-wide counter value taken at creation and at every checkpoint
	atomic<idx_t> epoch;
	unordered_map<idx_t, shared_ptr<ColumnDictionary>> columns;
	unordered_map<idx_t, shared_ptr<ColumnDictionary>> persisted_columns;
};
//! The table's registry entry (created when `create`), or null
shared_ptr<TableDictionaries> TableEntry(const shared_ptr<DataTableInfo> &info, bool create);
//! A checkpoint of the table: a new storage epoch (no-op without an entry)
void BumpEpoch(const shared_ptr<DataTableInfo> &info);
//! The table's entry erased (~DataTableInfo, and every DataTable alter constructor: the new table shares the
//! DataTableInfo and DROP COLUMN renumbers the storage indexes); its dictionaries are released outside the map's lock
void ReleaseTable(const DataTableInfo &info) noexcept;
//! The published dictionary of the table's storage column at the current epoch, built from the table's own row
//! groups, or null
shared_ptr<ColumnDictionary> Published(DataTable &table, idx_t storage_index);
//! The publication version: a process-wide counter that moves after every event that can change what Published()
//! returns for some column (a publish, a storage epoch bump, a table's entry released). PlanCodeKeys types a key only on
//! a published entry, so a plan kept beyond its statement (a client's plan cache) keys on this value read before
//! it plans: a plan made before a publish is then never reused after it, and one that typed a dictionary is never
//! reused once that dictionary stops being the column's published one
idx_t PublicationVersion();

//! The build trigger: for each admitted VARCHAR column among the scanned storage ids whose entry is unbuilt, build it
//! whole and publish (the calling thread; nobody waits; the entry refused on a reservation failure, or when a
//! reservation step would take every global-dictionary reservation - published dictionaries' and concurrent builds' -
//! above a quarter of memory_limit - the pressure refusal - or, before it starts, when the column's statistics estimate
//! cannot fit that budget)
void EnsureBuilt(ClientContext &context, DataTable &table, const vector<idx_t> &storage_ids);

//! Whether a vector is a dictionary vector over a published column dictionary (its id carries the dict_global- prefix
//! only the global dictionary's publish assigns)
bool IsPublishedVector(const Vector &vector);

//===--------------------------------------------------------------------===//
// Publication to grouping-key consumers and the memory gate
//===--------------------------------------------------------------------===//
//! The admission share: the scan's estimated post-filter cardinality below this share of the table's rows refuses the
//! build for that execution, for every column the scan marks, whatever the column's size (a tuning threshold;
//! see CLICKBENCH-FORK.md)
static constexpr double GATE_SHARE = 0.01;
//! the pre-build estimator of a column's full build reservation: its statistics' approx-unique count (the planner's
//! HyperLogLog, read at plan time without I/O) times this many bytes
static constexpr idx_t ESTIMATED_BYTES_PER_UNIQUE = 1024;

//! One table scan's publication, planned from the aggregate above it: the admitted columns an aggregate's group item or
//! DISTINCT argument resolves to (the planner's mark), each with the memory gate's decision for the scan's executions.
//! It travels with the scan's bind data (a TableScanBindData subclass the planner substitutes, its base fields copied as
//! TableScanBindData::Copy copies them), never as a process global: an unmarked scan of a published column builds nothing
//! and emits the standard per-segment vectors.
struct ScanPublication {
	struct Column {
		idx_t storage_index = 0;
		string name;
		//! the approx-unique count of the column's statistics and the estimated reservation derived from it
		idx_t approx_unique = 0;
		idx_t estimated_reserve = 0;
		//! the scan's estimated post-filter cardinality and the table's rows
		idx_t estimated_rows = 0;
		idx_t table_rows = 0;
		//! the memory gate refuses this column's build (and its emission) for the scan's executions
		bool gated = false;
		//! Every consumer above the scan reads only codes, through these stored translations and this entry; else null
		shared_ptr<PersistedTranslations> codes_only;
		shared_ptr<ColumnDictionary> codes_only_dict;
		//! set when the consumer decides at each execution whether the scan reads codes only: its sink stores the
		//! decision before the scan starts, and a refused execution reads strings for the generic path; null: every
		//! execution reads codes only
		shared_ptr<atomic<bool>> codes_only_admitted;
		//! the scan reads the column for its pushed filter only (MarkFilterOnlyCodes): nothing reads its values
		bool filter_only = false;
		//! the scan reads the column for a first-keys code group (MarkCodeGroupKey): its consumers read its codes only
		bool group_key_codes = false;

		//! the scan reads the column's codes only in this execution
		bool ReadsCodesOnly() const {
			return codes_only && (!codes_only_admitted || codes_only_admitted->load(std::memory_order_acquire));
		}
	};
	vector<Column> columns;

	//! the column is marked (a grouping-key or DISTINCT consumer) - gated or not
	optional_ptr<const Column> Find(idx_t storage_index) const;
	//! the column is marked and not gated: the scan builds it (once) and emits it over the published child
	bool Publishes(idx_t storage_index) const;
};
//! The publication of the table scan running on this thread (set by the scan around its storage calls), or null (an
//! unmarked scan: nothing built, nothing emitted over a global dictionary)
const ScanPublication *&ThreadPublication();
//! Sets ThreadPublication() for one scope and restores the previous value on exit
struct ThreadPublicationScope {
	explicit ThreadPublicationScope(const ScanPublication *publication);
	~ThreadPublicationScope();
	ThreadPublicationScope(const ThreadPublicationScope &) = delete;
	ThreadPublicationScope &operator=(const ThreadPublicationScope &) = delete;

private:
	const ScanPublication *previous;
};
//! The publication a seq_scan's bind data carries, or null (an unmarked scan)
shared_ptr<ScanPublication> ScanPublicationOf(const FunctionData *bind_data);
//! Plan time: mark on the child's table scan (through PROJECTION* / FILTER*) the admitted VARCHAR columns the
//! aggregate's group items and DISTINCT aggregate arguments resolve to, with the memory gate's decision per column (the
//! scan's bind data is replaced by the publishing subclass on its first mark)
void MarkKeyConsumers(ClientContext &context, PhysicalOperator &child, const vector<unique_ptr<Expression>> &groups,
                      const vector<unique_ptr<Expression>> &aggregates);
//! The segment's published translation for the scan on this thread: FindTranslation when the thread's publication
//! publishes the translation's column, else null
shared_ptr<SegmentTranslation> FindScanTranslation(ColumnSegment &segment);

//! The group column a hash table append borrows on this thread (its strings stored verbatim, no heap), or
//! DConstants::INVALID_INDEX; set only around that append
idx_t &ThreadBorrowColumn();

//===--------------------------------------------------------------------===//
// The code keys of one physical hash aggregate
//===--------------------------------------------------------------------===//
struct CodeKey {
	//! the group index and the input chunk column it references
	idx_t group_index;
	idx_t chunk_index;
	shared_ptr<ColumnDictionary> dict;
	//! the operator emits this key's codes (INTEGER) and a projection above its Top-N decodes them (PlanLateDecode)
	bool emit_codes = false;
};

//! The conversion of one typed aggregate's input and output
class CodeKeys {
public:
	vector<CodeKey> keys;
	//! the operator's input chunk types with each typed column INTEGER
	vector<LogicalType> input_types;
	//! the operator's output types with each typed group INTEGER
	vector<LogicalType> internal_output_types;
	//! overflow codes for strings absent from the dictionary (a vector not over the child), per key
	struct Overflow {
		mutex lock;
		unordered_map<string, uint32_t> codes;
		vector<string> strings;
	};
	vector<unique_ptr<Overflow>> overflow;
	string column_names;

	//! Convert the input chunk: every typed column to an INTEGER code vector (the lookup path is taken for a vector
	//! not over the child)
	void ConvertInput(DataChunk &input, DataChunk &converted, vector<Vector> &code_vectors) const;
	//! Convert an internal output chunk (codes) into the operator's output chunk (strings over the pinned child)
	void ConvertOutput(DataChunk &internal, DataChunk &output) const;
};

//! Plan time: type the published admitted group keys of a grouped hash aggregate as INTEGER codes. `groups` are the
//! aggregate's BOUND_REF groups over `child`; returns null when nothing is typed.
unique_ptr<CodeKeys> PlanCodeKeys(ClientContext &context, PhysicalOperator &child,
                                  vector<unique_ptr<Expression>> &groups, const vector<LogicalType> &output_types,
                                  const vector<unique_ptr<Expression>> &aggregates, idx_t grouping_set_count);
//! Plan time: the published dictionary of the admitted, uncollated VARCHAR column that input column `chunk_index`
//! of an operator over `child` resolves to (PROJECTION* / FILTER* over one seq_scan, BOUND_REFs only), when the scan
//! publishes that column for this plan (PlanCodeKeys's conditions for one column); else null. `table` is set to the
//! scanned table's storage when a dictionary is returned
shared_ptr<ColumnDictionary> PlanPublishedColumn(PhysicalOperator &child, idx_t chunk_index, const LogicalType &type,
                                                 optional_ptr<DataTable> &table);

//! The publication entry of a column read through stored translations at the current epoch (segments linked), or null
shared_ptr<ColumnDictionary> PublishPersisted(DataTable &table, idx_t storage_index);
//! The stored translations a codes-only tag child's id names, or null
shared_ptr<PersistedTranslations> CodesOnlyTranslationsOf(const string &child_id);
//! Whether every table scan of the plan that reads a column codes-only through stored translations may still execute
//! as planned: a column read codes-only at every execution still has the plan's publication entry (none while an
//! append lies outside the translations or the column carries an update; a checkpoint starts a new one) and this
//! transaction holds no local storage of the table; a column whose consumer decides at each execution still has the
//! plan's translations (a checkpoint replaced them otherwise). False when a scan's does not
bool PlanTranslationsCurrent(ClientContext &context, const PhysicalOperator &root);
//! Plan time, a plan that reads a column through its stored translations: the client re-binds such a prepared
//! statement whose translations went stale before it executes (a ClientContextState)
void NoteStoredTranslationPlan(ClientContext &context);
//! The user error of a plan that reads stale stored translations (a statement planned before a checkpoint that changed
//! them, executed without a re-bind)
[[noreturn]] void ThrowStaleTranslations(const string &column_name);

//! Plan time, a Top-N over PROJECTION* over a grouped hash aggregate whose code keys include a column read through its
//! stored translations: each such key that the Top-N does not order by and the projections pass through is emitted as
//! its codes (the aggregate's and the projections' types become INTEGER there); returns the Top-N output positions to
//! decode, each with the key's code keys (empty: nothing changed)
struct LateDecode {
	idx_t position;
	shared_ptr<CodeKeys> keys;
	idx_t key_index;
};
vector<LateDecode> PlanLateDecode(PhysicalOperator &child, const vector<BoundOrderByNode> &orders);
//! The decode of one late-decoded key: its codes (INTEGER) to strings
unique_ptr<Expression> LateDecodeExpression(const LateDecode &decode, idx_t index, const LogicalType &type);
//! Plan time, the fused bitmap class over a column read through its stored translations: the scan emits that column's
//! codes only in the executions `admitted` holds true for (its sink stores the decision before the scan starts)
void MarkCodesOnly(PhysicalOperator &child, idx_t chunk_index, const shared_ptr<ColumnDictionary> &dict,
                   shared_ptr<atomic<bool>> admitted);
//! Plan time, a seq_scan of a DuckDB table: each uncollated VARCHAR column the scan reads for its pushed filter only (not
//! emitted), where that filter is decided on codes (CodeTranslatable), is read codes only at every execution through its
//! stored translations (PublishPersisted's conditions; the scan estimated at kStoredCodeKeysMinScanRows rows or more and
//! this transaction holding no local storage of the table; kFilterOnlyCodesOnly)
void MarkFilterOnlyCodes(ClientContext &context, PhysicalTableScan &scan);
//! The columns a seq_scan's bind data reads codes only through stored translations, one per line, "(filter only)" after a
//! column read for its filter alone, "(group key)" after a first-keys code group, or empty (EXPLAIN)
string CodesOnlyColumnNames(const FunctionData *bind_data);
//! Optimizer time, a first-keys code group (FirstKeysAggregate, kFirstKeysCodeKeys): marks column `column_index` (an
//! index into the get's column ids) of a seq_scan's logical get gated and read codes only at every execution through
//! `dict`'s stored translations, on the publication its bind data carries - substituted on the first mark, and shared
//! by every copy of that bind data, so a copy of the get made after the mark reads the same codes. False (nothing
//! marked) for a get that already carries dynamic filters, a get that is not a seq_scan of a DuckDB table reading the
//! column once, or a dictionary without stored translations
bool MarkCodeGroupKey(ClientContext &context, LogicalGet &get, idx_t column_index,
                      const shared_ptr<ColumnDictionary> &dict);
//! __dict_global_codes(child): the codes (INTEGER) of a VARCHAR column read codes only, NULL where the code is 0 (the
//! NULL string's); any vector other than a codes-only one (a NULL constant aside) throws an InternalException
unique_ptr<Expression> CodesExpression(unique_ptr<Expression> child);
//! __dict_global_decode(child): codes (INTEGER) of `dict`'s stored translations to strings of `type`, NULL for code 0
//! (the late decode of one key, LateDecodeExpression's function, over a logical child)
unique_ptr<Expression> CodeDecodeExpression(const shared_ptr<ColumnDictionary> &dict, unique_ptr<Expression> child,
                                            const LogicalType &type);

//! The code keys of a physical hash aggregate, kept beside the operator (keyed by its address), never inside it
void RegisterCodeKeys(const void *op, shared_ptr<CodeKeys> keys);
shared_ptr<CodeKeys> FindCodeKeys(const void *op);
void ReleaseCodeKeys(const void *op);

} // namespace dict_global
} // namespace duckdb
