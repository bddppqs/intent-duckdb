//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/aggregate/fused_integer_aggregate.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/enums/order_preservation_type.hpp"
#include "duckdb/common/enums/tuple_data_layout_enums.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/string_map_set.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/execution/progress_data.hpp"
#include "duckdb/storage/arena_allocator.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"

namespace duckdb {
class BufferManager;
class ClientContext;
class DataTable;
class ExecutionContext;
class PhysicalHashAggregate;
class PhysicalOperator;
class PhysicalUngroupedAggregate;
namespace dict_global {
class ColumnDictionary;
}

//! How one aggregate of the fused path is computed from a group's states (every input is non-NULL: the gate proved it);
//! DISTINCT_COUNT reads the group's number of distinct (g, x) entries
enum class FusedAggregateKind : uint8_t { COUNT, SUM, AVG, DISTINCT_COUNT };

//! One aggregate's output: COUNT reads the group's row count; SUM and AVG read one int64 sum state (for the mixed
//! DISTINCT shape: sum_index is the output's index among the summed or averaged companion inputs)
struct FusedAggregateOutput {
	FusedAggregateKind kind;
	idx_t sum_index;
};

//! One column of the compact row: a group key or an aggregate input, stored as its own bytes
struct FusedRowColumn {
	//! Column of the aggregate's input chunk
	idx_t chunk_index;
	//! Byte offset inside the compact row
	idx_t offset;
	//! Byte width: 1, 2, 4 or 8
	idx_t width;
	//! a VARCHAR key stored as its 2-byte gid in the operator's string -> gid map
	bool gid = false;
};

class FusedAggregateGlobalState;
class FusedAggregateLocalState;

//! The fused integer aggregate. For a GROUP BY over one or two non-NULL integer keys with count / sum /
//! avg over non-NULL integers (the shape checks of TryAttach), phase 1 (Sink) appends each input row as a compact row
//! {keys, inputs} into one of 4096 partitions through a per-thread 64-byte write-combining line per partition, and
//! phase 2 (GetData) builds one table per partition, sized once from the partition's row count, and scans it once: no
//! Combine of partial tables, no repartition and no second aggregation. A NULL (a plan the data has outrun), a memory
//! budget crossing or a failed allocation drains every buffered row into the radix path exactly (the drain),
//! after which the operator runs the general code path.
//! The DISTINCT class: count(DISTINCT x) over one non-NULL integer x with zero to two non-NULL integer keys g
//! (and zero to three of the non-distinct aggregates above) takes the same path with the compact row keyed by (g, x);
//! phase 1 folds adjacent equal keys inside one input chunk when there is no non-distinct aggregate, and phase 2 folds
//! each partition's (g, x) table into a per-task group table, merged once by the last task (GetData). A drain re-sinks
//! through the operator's own Sink body (SinkDistinct included).
//! The mixed DISTINCT shape: the class attaches beside one to three non-distinct aggregates too (none reading x),
//! every attached shape folds, and the non-distinct aggregates are computed by a per-thread g-keyed companion table over
//! every input row; a drain re-sinks the (g, x) rows to the distinct table alone and re-expands each companion entry into
//! value-split rows for the regular sink alone.
//! The run kind: on a plan the grouped run channel takes (one integer key, COUNT(*) only), the kernel attaches
//! beside the run descriptor: the scan's run batches arrive through SinkRuns as compact rows {key, run length}, the
//! vectors the run branch declines through Sink with run length 1, phase 2 folds count += run length, and the drain
//! re-sinks every buffered row through the radix path's SinkRuns with its run length.
class FusedIntegerAggregate {
public:
	//! Partitions of phase 1: the top 12 bits of the row hash
	static constexpr idx_t PARTITION_BITS = 12;
	static constexpr idx_t PARTITION_COUNT = idx_t(1) << PARTITION_BITS;
	//! One write-combining line per partition and thread
	static constexpr idx_t LINE_BYTES = 64;
	//! Partition chunks: a {next, rows} header, then packed compact rows
	static constexpr idx_t CHUNK_BYTES = 4096;
	static constexpr idx_t CHUNK_HEADER_BYTES = 16;
	//! Before any row of an input chunk is appended, the thread's pool holds at least this many free chunks: each
	//! appended row takes at most one chunk (its partition's first chunk, or the next one when a flush fills the current)
	static constexpr idx_t POOL_MINIMUM_FREE_CHUNKS = STANDARD_VECTOR_SIZE;
	//! Free chunks per slab: one buffer-manager allocation of 63 x 4 KiB + 64 bytes of alignment slack (256 KiB with its
	//! header) holds 63 64-byte-aligned chunks, so a top-up to POOL_MINIMUM_FREE_CHUNKS takes at most 33 slabs and
	//! leaves at most 2047 + 63 = 2110 free chunks
	static constexpr idx_t CHUNKS_PER_SLAB = 63;
	//! the per-thread charge, 24.25 MiB: the partition-chunk floor (4096 x 4 KiB = 16 MiB) plus the pool's ceiling
	//! (8.25 MiB)
	static constexpr idx_t PER_THREAD_CHARGE_BYTES = idx_t(97) * 1024 * 1024 / 4;
	//! the row floor and the per-row charge of the estimate
	static constexpr idx_t MINIMUM_ESTIMATED_ROWS = idx_t(1) << 20;
	static constexpr idx_t ESTIMATE_ROW_BYTES = 32;
	//! the DISTINCT class's group estimate ceiling (the per-task group tables are sized from it)
	static constexpr idx_t MAXIMUM_ESTIMATED_GROUPS = idx_t(1) << 20;
	//! a VARCHAR key's base-statistics distinct-count estimate at most this, and at most this many
	//! distinct strings mapped at run time (the next one is a crossing: the drain)
	static constexpr idx_t MAXIMUM_GIDS = 65535;
	//! the ceilings on key bytes and compact-row bytes (the grouped class's key bytes; see kFusedSixteenByteKeys)
	static constexpr idx_t MAXIMUM_KEY_BYTES = 16;
	static constexpr idx_t MAXIMUM_ROW_BYTES = 32;
	static constexpr idx_t MAXIMUM_AGGREGATES = 4;
	//! Phase 2: capacity next_pow2(2 x rows) in [1024, 2^21] entries, sized once; 16-row hash-and-prefetch batches
	static constexpr idx_t TABLE_MINIMUM_CAPACITY = 1024;
	static constexpr idx_t TABLE_MAXIMUM_CAPACITY = idx_t(1) << 21;
	static constexpr idx_t TABLE_BATCH = 16;
	//! the stored hash bits appended to the compact row (the 32 hash bits below the partition bits)
	static constexpr idx_t STORED_HASH_BYTES = 4;
	//! The bitmap class: the bitmap words one phase-2 stripe covers (8 KiB of each thread's bitmap)
	static constexpr idx_t BITMAP_STRIPE_WORDS = 1024;

	FusedIntegerAggregate();

	//! The shape checks: attaches a fused path to the aggregate when every condition holds; a
	//! DISTINCT shape takes the DISTINCT class's checks
	static void TryAttach(ClientContext &context, PhysicalHashAggregate &op, TupleDataValidityType group_validity);
	//! the DISTINCT class's checks on the ungrouped aggregate (no keys)
	static void TryAttachUngrouped(ClientContext &context, PhysicalUngroupedAggregate &op);

	unique_ptr<FusedAggregateGlobalState> GetGlobalSinkState(ClientContext &context) const;
	unique_ptr<FusedAggregateLocalState> GetLocalSinkState(ExecutionContext &context) const;
	//! True: the chunk is in the fused buffers. False: the operator is abandoned and this thread has drained its
	//! buffered rows, so the caller sinks the chunk through the regular Sink body
	bool Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input, const PhysicalOperator &op,
	          FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) const;
	//! True: the thread's partition lists were handed over. False: the operator is abandoned and this thread has
	//! drained its lists, so the caller runs the regular Combine
	bool Combine(ExecutionContext &context, OperatorSinkInput &input, const PhysicalOperator &op,
	             FusedAggregateGlobalState &gstate, FusedAggregateLocalState &lstate) const;
	//! True: phase 2 owns the source. False: the operator is abandoned, so the caller runs the regular Finalize
	bool Finalize(FusedAggregateGlobalState &gstate) const;
	//! run kind: one batch of (key, run length) pairs from the grouped run channel's sink. True: the batch is in
	//! the fused buffers. False: the operator is abandoned and this state has drained its buffered rows, so the caller
	//! sinks the batch through the radix path's SinkRuns
	bool SinkRuns(ClientContext &context, Vector &values, const uint16_t *counts, idx_t run_count,
	              const PhysicalHashAggregate &op, FusedAggregateGlobalState &gstate,
	              FusedAggregateLocalState &lstate) const;
	//! run kind: Combine without an ExecutionContext (a run partial's, at the operator's Finalize). True: the
	//! lists were handed over. False: the operator is abandoned and the lists were drained into the radix state
	bool CombineRuns(ClientContext &context, const PhysicalHashAggregate &op, FusedAggregateGlobalState &gstate,
	                 FusedAggregateLocalState &lstate) const;
	//! the set member's run form: one batch of (x, run length) pairs from the run descriptor's set form, one row {x}
	//! per run. True: the batch is in the fused buffers. False: the operator is abandoned and this state has drained its
	//! buffered rows into the distinct table, so the caller sinks the batch there
	bool SinkSetRuns(ClientContext &context, Vector &values, const uint16_t *counts, idx_t run_count,
	                 const PhysicalUngroupedAggregate &op, FusedAggregateGlobalState &gstate,
	                 FusedAggregateLocalState &lstate) const;
	//! the set member's run form: a run partial's Combine (at its scan's end or the operator's Finalize). True: the
	//! lists were handed over. False: the operator is abandoned and the lists were drained into the distinct table
	bool CombineSetRuns(ClientContext &context, const PhysicalUngroupedAggregate &op, FusedAggregateGlobalState &gstate,
	                    FusedAggregateLocalState &lstate) const;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context, FusedAggregateGlobalState &gstate) const;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context) const;
	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, FusedAggregateGlobalState &gstate,
	                         OperatorSourceInput &input) const;
	ProgressData GetProgress(FusedAggregateGlobalState &gstate, GlobalSourceState &source_state) const;

	//! The EXPLAIN text: "keys=n aggregates=n partitions=4096 budget_mib=B", then the counters when a sink state exists; the
	//! DISTINCT class adds "distinct=1", source_order (the operator's SourceOrder()) and folded_rows
	string ParamsString(optional_ptr<FusedAggregateGlobalState> gstate, OrderPreservationType source_order) const;
	//! the DISTINCT class's phase-2 counters under "Fused Distinct Phase 2", once the merge is done (empty before)
	InsertionOrderPreservingMap<string> ExtraSourceParams(GlobalSourceState &source_state) const;

public:
	//! The compact row: the keys first (group order), then the distinct aggregate inputs
	vector<FusedRowColumn> columns;
	idx_t key_count;
	//! Bytes of the key region at the start of the row (at most 12)
	idx_t key_bytes;
	//! Row width, padded to a multiple of 4 and at least 8 (at most 32)
	idx_t row_width;
	//! One per aggregate, in the aggregate order
	vector<FusedAggregateOutput> outputs;
	//! Per sum state: the row column (an aggregate input) it sums
	vector<idx_t> sum_columns;
	//! The aggregate's input chunk types (the rebuilt chunks of a drain)
	vector<LogicalType> input_types;
	//! The operator's output types (keys, then aggregates)
	vector<LogicalType> output_types;
	//! B = GetMaxMemory() / 4 at plan time, in bytes: the drain threshold unless a compile-time override is set
	idx_t budget_bytes;
	//! the DISTINCT class. The key region is (g..., x): columns[0, key_count) are the keys g, columns[key_count]
	//! is x, and key_bytes covers both; group_bytes covers g alone (0 for the ungrouped form)
	bool distinct;
	idx_t group_bytes;
	//! phase 1 folds adjacent equal keys inside one input chunk (every attached DISTINCT shape)
	bool fold;
	//! the mixed DISTINCT shape (a non-distinct aggregate beside the distinct one). The compact row stays (g..., x)
	//! and sum_columns stays empty: the non-distinct inputs live in the companion input list (chunk_index and width per
	//! summed, averaged or counted input; offset unused), and companion_sums lists, per sum state, the companion input it
	//! sums
	bool mixed;
	vector<FusedRowColumn> companion_inputs;
	vector<idx_t> companion_sums;
	//! the number of VARCHAR keys admitted as gids (the DISTINCT class only; 0 without)
	idx_t gid_keys;
	//! the stored hash (both classes): phase 1 stores the 32 hash bits below the partition bits at hash_offset, the row
	//! widened by STORED_HASH_BYTES (a shape whose row would pass MAXIMUM_ROW_BYTES keeps the plain row),
	//! and phase 2 reads them instead of recomputing the hash
	bool hash_stored;
	idx_t hash_offset;
	//! the chained build (needs the stored hash): phase 2 links each partition's rows in place into an exact directory
	//! with an append-only group array (no open-addressing table, no key copy) and the output scans the groups
	bool chain;
	//! the run kind (fed by the grouped run channel); the compact row is {key, uint32_t run length}
	bool run_kind;
	//! run kind: byte offset of the run length in the compact row (right after the key bytes); with the last-key fold,
	//! of the folded row's count (right after the inputs)
	idx_t run_length_offset;
	//! the last-key fold (kFusedLastKeyFold): a COUNT-only shape of the grouped class carries a uint32_t count, phase 1
	//! folds each row whose key equals its partition's last row into that row's count, phase 2 adds counts, and the
	//! drain re-sinks a folded row as its count's rows
	bool last_key_fold;
	//! the bitmap class (kFusedDistinctBitmap). An ungrouped count(DISTINCT x), the operator's one aggregate, whose x is
	//! an uncollated VARCHAR column the global dictionary publishes for the plan, takes one bitmap of the published dictionary's
	//! codes per thread: phase 1 sets each row's code bit (a vector over the published child reads its codes from its
	//! selection; any other vector looks each string up, and a string the dictionary does not hold goes to the thread's
	//! overflow set), phase 2 ORs the threads' bitmaps stripe by stripe across the source tasks and counts the bits, code 0
	//! (the NULL slot) excluded, plus the distinct overflow strings. No hash, no partition, no reservation, no drain.
	//! columns[0] is x (chunk_index alone); bitmap_words = ceil(codes / 64)
	bool bitmap;
	//! the set member of the DISTINCT class (kFusedDistinctSet). An ungrouped count(DISTINCT x), the operator's one
	//! aggregate, over one integer x: the compact row is x alone (no stored hash), and phase 2 counts each partition's
	//! distinct x by inserting it into a set of the key word alone, a new key counted at its insert - no (g, x) entry
	//! count, no group table, no scan of the set; the task that finishes last emits the sum
	bool distinct_set = false;
	//! the set member's run form (kFusedRunFedDistinctSet): the run descriptor's set form feeds the scan's runs of x
	//! through SinkSetRuns as the set member's compact row x alone, one row per run; the vectors the run branch declines
	//! arrive through Sink, and a run-fed state's drain re-sinks its rows into the distinct table through the radix path's
	//! SinkRuns
	bool set_runs = false;
	shared_ptr<dict_global::ColumnDictionary> bitmap_dict;
	idx_t bitmap_words;
	//! the coverage bound: the scanned table's storage, re-read at every execution (GetGlobalSinkState): the
	//! transaction's local storage, the column's publication (still bitmap_dict), its coverage (rows_covered ==
	//! GetTotalRows()), its updates and the memory bound at the executing threads and memory limit; a failure abandons
	//! the execution to the generic path (a plain pointer: the const operator re-reads the table's mutable state, as
	//! the plan's scan does)
	DataTable *bitmap_table = nullptr;
	//! a column read through stored translations: whether this execution's scan reads its codes only (set by
	//! GetGlobalSinkState to the execution's admission; the scan reads it at its initialisation), else null
	shared_ptr<atomic<bool>> bitmap_codes_only;
};

//! A group table of the DISTINCT class, keyed by the key words of g alone: entries {g words, distinct, count, one
//! int64 sum state per summed or averaged input}, occupied iff distinct != 0 (the ungrouped form: one entry, no key
//! words). The companion has the same layout, its distinct word the occupancy marker alone (never read as a count)
struct FusedGroupTable {
	BufferHandle handle;
	uint64_t *entries = nullptr;
	idx_t capacity = 0;
	idx_t occupancy = 0;
};

//! A thread's partition lists after its Combine: every chunk it filled, the slabs that hold them, per-partition rows
//! (and the thread's companion table, which travels with its lists)
struct FusedThreadPartitions {
	vector<data_ptr_t> heads;
	vector<idx_t> rows;
	vector<BufferHandle> slabs;
	unique_ptr<FusedGroupTable> companion;
};

//! Which reservation refused the first crossing (NONE when none did)
enum class FusedCrossing : uint8_t { NONE, ROWS, COMPANION_GROWTH, OOM };

class FusedAggregateGlobalState {
public:
	explicit FusedAggregateGlobalState(BufferManager &buffer_manager);

	BufferManager &buffer_manager;
	//! Guards abandoned and handed_over (never the radix state's lock, which the standard Sink re-takes)
	mutex lock;
	//! Set once, by the first crossing; from then on the operator runs the standard code
	atomic<bool> abandoned;
	//! Partition lists handed over by Combines before any crossing
	vector<unique_ptr<FusedThreadPartitions>> handed_over;
	//! Drain threshold in rows' bytes (B)
	idx_t drain_threshold;

	//! The EXPLAIN ANALYZE counters
	atomic<idx_t> input_rows;
	//! rows the fold never buffered
	atomic<idx_t> folded_rows;
	//! input rows added to the companion tables
	atomic<idx_t> companion_rows;
	//! the reservation that refused the first crossing (a FusedCrossing), set once by the thread that abandons
	atomic<uint8_t> crossing;
	atomic<idx_t> drained_rows;
	atomic<idx_t> combines_taken;
	atomic<idx_t> reserved_bytes;
	atomic<idx_t> slabs_allocated;
	atomic<idx_t> partitions_nonempty;
	//! the stored hash's and the chained build's counters (EXPLAIN ANALYZE's), each accumulated per input chunk or
	//! per partition, never per row: rows appended with stored bits, rows linked by the chained build and its groups,
	//! the open-addressing probes of the stored-bits build, and the partitions built
	atomic<idx_t> hash_stored_rows;
	atomic<idx_t> chain_rows;
	atomic<idx_t> chain_new;
	atomic<idx_t> table_probes;
	atomic<idx_t> partitions_built;
	//! run kind: the run rows appended through SinkRuns and the sum of their run lengths (EXPLAIN ANALYZE counters)
	atomic<idx_t> run_rows;
	atomic<idx_t> run_length_sum;
	//! the set member's run form: the rows the run-fed states' drains re-sank (EXPLAIN ANALYZE counter)
	atomic<idx_t> run_drained;
	//! run kind: the radix global sink state, the drain's target (the set member's run form: the distinct table's)
	optional_ptr<GlobalSinkState> run_radix_global;

	//! Phase 2 (set by Finalize): rows per partition over every handed-over list
	vector<idx_t> partition_rows;
	bool finalized;

	//! the string -> gid map of the VARCHAR keys, insertion only, under gid_lock; each string's bytes copied
	//! into gid_arena (whose blocks never move, never reset while the operator lives), gid_strings[gid] its string
	mutex gid_lock;
	ArenaAllocator gid_arena;
	string_map_t<uint16_t> gid_of;
	vector<string_t> gid_strings;
	//! the EXPLAIN ANALYZE gid counters: strings mapped, per-thread dictionary tables built, rows mapped one by one
	atomic<idx_t> gid_count;
	atomic<idx_t> gid_maps;
	atomic<idx_t> gid_rows_flat;

	//! The bitmap class: the threads' bitmaps (pinned buffer-manager blocks of bitmap_words words), the union of their overflow
	//! strings and the rows they looked up (a vector not over the published child), handed over at Combine under `lock`
	vector<BufferHandle> bitmaps;
	unordered_set<string> bitmap_overflow;
	idx_t bitmap_lookups = 0;
	//! The coverage bound: this execution's re-check (GetGlobalSinkState) - engaged, or refused with its reason (the execution is
	//! abandoned to the generic path); and the set bits phase 2 read at or beyond the code count in the last word
	bool bitmap_engaged = false;
	uint8_t bitmap_refusal = 0;
	atomic<idx_t> bitmap_padding_bits {0};
};

//! One thread's gids of one VARCHAR key column - the dictionary whose code -> gid table it caches (by
//! DictionaryVector::DictionaryId), the table, and one input chunk's gids
struct FusedGidCache {
	string dictionary_id;
	vector<uint16_t> table;
	unsafe_unique_array<uint16_t> rows;
};

class FusedAggregateLocalState {
public:
	FusedAggregateLocalState(ExecutionContext &context, const FusedIntegerAggregate &fused);
	//! A run partial's state, created on a scan thread without an ExecutionContext
	explicit FusedAggregateLocalState(const FusedIntegerAggregate &fused);

	//! Write-combining lines (PARTITION_COUNT x LINE_BYTES) and their fill (rows per line)
	unsafe_unique_array<data_t> lines;
	unsafe_unique_array<uint8_t> fill;
	//! Per partition: the next free byte of the current chunk, lines of room left in it, the current chunk, the head
	unsafe_unique_array<data_ptr_t> cursor;
	unsafe_unique_array<uint16_t> room;
	unsafe_unique_array<data_ptr_t> tail;
	unsafe_unique_array<data_ptr_t> head;
	//! Free chunks carved from the thread's slabs, and the slabs (buffer-manager blocks, pinned)
	vector<data_ptr_t> pool;
	vector<BufferHandle> slabs;
	//! One input chunk's compact rows, their hashes and the unified formats of the row's columns
	unsafe_unique_array<data_t> row_buffer;
	unsafe_unique_array<hash_t> hashes;
	vector<UnifiedVectorFormat> formats;

	//! Lists taken over from earlier Combines by this thread's crossing
	vector<unique_ptr<FusedThreadPartitions>> taken;
	//! Set once this thread has re-sunk its buffered rows; every later chunk goes through the regular Sink
	bool drained;
	//! Set while the drain re-sinks rebuilt chunks through the operator's Sink (they are not input rows)
	bool resinking;
	//! while a mixed shape's drain re-sinks, the rebuilt (g, x) chunks go to the distinct table alone
	//! (resinking_distinct_only) and the re-expanded companion rows to the regular sink alone (resinking_regular_only)
	bool resinking_distinct_only;
	bool resinking_regular_only;
	//! the thread's companion table (null until its first pre-grow), the unified formats of the companion inputs,
	//! g's bytes of one input chunk (16 bytes per row, at the compact row's g offsets) and one entry pointer per row
	unique_ptr<FusedGroupTable> companion;
	vector<UnifiedVectorFormat> companion_formats;
	unsafe_unique_array<data_t> group_rows;
	unsafe_unique_array<uint64_t *> companion_entries;
	//! The rebuilt chunk of a drain (the aggregate's input chunk types; for the run kind the one key column)
	DataChunk drain_chunk;
	//! per compact-row column (used for the gid columns alone)
	vector<FusedGidCache> gid_caches;
	//! run kind: the radix local sink state this state drains into (the set member's run form: the distinct table's), and
	//! one drain batch's run lengths
	optional_ptr<LocalSinkState> run_radix_local;
	unsafe_unique_array<uint16_t> drain_counts;
	//! The bitmap class: the thread's bitmap (allocated at its first chunk), its overflow strings, its input rows and the rows it
	//! looked up
	BufferHandle bitmap_handle;
	uint64_t *bitmap = nullptr;
	unordered_set<string> bitmap_overflow;
	idx_t bitmap_rows = 0;
	idx_t bitmap_lookups = 0;
};

} // namespace duckdb
