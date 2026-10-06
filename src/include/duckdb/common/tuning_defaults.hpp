//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/tuning_defaults.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

namespace duckdb {

//! Build-time feature flags and tuning constants of this fork. Each flag enables one mechanism; with a flag set to
//! false, the code takes the path it would take without that mechanism. Only the per-architecture block below
//! differs by platform.

//===--------------------------------------------------------------------===//
// Per-architecture defaults
//===--------------------------------------------------------------------===//
#if defined(__aarch64__)
//! The global dictionary's codes feed the fused DISTINCT kernel
static constexpr bool kGlobalDictionaryFusedDistinct = false;
//! The fused integer aggregate admits a dictionary-coded VARCHAR group key
static constexpr bool kFusedIntegerAggregateVarcharKeys = false;
//! A new VARCHAR group key whose bytes live in a storage dictionary is appended without a heap copy
static constexpr bool kBorrowedStringGroupKeys = false;
#else
//! The global dictionary's codes feed the fused DISTINCT kernel
static constexpr bool kGlobalDictionaryFusedDistinct = true;
//! The fused integer aggregate admits a dictionary-coded VARCHAR group key
static constexpr bool kFusedIntegerAggregateVarcharKeys = true;
//! A new VARCHAR group key whose bytes live in a storage dictionary is appended without a heap copy
static constexpr bool kBorrowedStringGroupKeys = true;
#endif

//===--------------------------------------------------------------------===//
// Aggregation
//===--------------------------------------------------------------------===//
//! The specialized aggregation kernel for groups keyed by at most two integer columns packed into 12 bytes or fewer
//! (three keys and 16 bytes with kFusedSixteenByteKeys)
static constexpr bool kFusedIntegerAggregate = true;
//! The kernel's grouped class takes up to three keys packed into 16 bytes or fewer (the DISTINCT class keeps two keys
//! and 12 bytes); the global dictionary's wide-key guard follows the same width
static constexpr bool kFusedSixteenByteKeys = true;
//! The kernel admits a group key computed from one column by a deterministic scalar function whose statistics bound
//! its range (for example minute(ts) or x * 2), compressed like a column key
static constexpr bool kFusedComputedGroupKeys = true;
//! The fused kernel's phase 1 folds a COUNT-only row whose key equals its partition's last row into that row's count
static constexpr bool kFusedLastKeyFold = true;
//! The last-key fold is refused when its count widens the compact row and, over an unfiltered table scan, the key
//! columns' distinct-count statistics leave too few repeated rows for folding them to pay for the wider row
static constexpr bool kFusedLastKeyFoldStatisticsGuard = true;
//! The kernel's DISTINCT class
static constexpr bool kFusedDistinctAggregate = true;
//! An ungrouped count(DISTINCT x) over a VARCHAR column the global dictionary publishes counts the dictionary codes in
//! one bitmap per thread (the bitmap class), when every row of the column is covered by the published dictionary
static constexpr bool kFusedDistinctBitmap = true;
//! An ungrouped count(DISTINCT x) over an integer column, the operator's one aggregate, counts each partition's distinct x
//! in a set of x alone (the set member): no stored hash, no per-partition group fold, no table scan
static constexpr bool kFusedDistinctSet = true;
//! The DISTINCT class folds duplicate keys while sinking
static constexpr bool kFusedDistinctFold = true;
//! The grouped DISTINCT class merges its group tables as a tree as the tasks finish
static constexpr bool kFusedDistinctTreeMerge = true;
//! The kernel's mixed DISTINCT shape (one distinct count beside plain aggregates)
static constexpr bool kFusedMixedDistinct = true;
//! The kernel accepts run-length encoded input
static constexpr bool kFusedRunChannel = true;
//! The kernel stores each row's hash for the partition build
static constexpr bool kFusedStoredHash = true;
//! The partition build chains on the stored hash bits
static constexpr bool kFusedStoredHashChain = true;
//! A chunk's bytes are reserved by one fetch_add, rolled back when the sum crosses the drain threshold
static constexpr bool kFusedAggregateAtomicReserve = true;
//! Each local state reserves from its own allowance, refilled from the shared budget
static constexpr bool kFusedAggregateLocalAllowance = true;
//! A run partial hands its state over at its seal
static constexpr bool kRunPartialSeal = true;
//! A run partial builds its state outside the shared lock
static constexpr bool kRunPartialLocalBuild = true;
//! The radix merge appends a new group without re-copying its string key
static constexpr bool kBorrowedGroupKeysInCombine = true;
//! MIN/MAX over strings keep their state in the aggregate's arena
static constexpr bool kMinMaxStringArena = true;

//===--------------------------------------------------------------------===//
// Global dictionary
//===--------------------------------------------------------------------===//
//! A scan builds and publishes a column-wide dictionary of a VARCHAR column
static constexpr bool kGlobalDictionary = true;
//! Aggregates group by global dictionary codes instead of strings
static constexpr bool kGlobalDictionaryCodeKeys = true;
//! Code keys are refused when the packed group width exceeds the fused kernel's key bytes (16 with
//! kFusedSixteenByteKeys, 12 otherwise)
static constexpr bool kGlobalDictionaryWideKeyGuard = true;
//! A scan with a string predicate whose selectivity cannot be estimated does not build the global dictionary
static constexpr bool kStringPredicateBuildGuard = true;
//! Each segment translation holds its own handle on the published dictionary child
static constexpr bool kSegmentTranslationChildHandle = true;
//! A segment remembers its translation lookup
static constexpr bool kSegmentTranslationCache = true;

//===--------------------------------------------------------------------===//
// Dictionary-compressed string scans and filters
//===--------------------------------------------------------------------===//
//! Per-dictionary filter verdicts are kept across scans
static constexpr bool kFilterVerdictCache = true;
//! The verdict cache also serves `=` / `<>` constant comparisons
static constexpr bool kConstantFilterVerdictCache = true;
//! The most MiB of verdict-slot weight alive at once
static constexpr int64_t kFilterVerdictCacheCapacityMiB = 64;
//! A vector read across segment pieces takes its codes' verdicts instead of evaluating each row
static constexpr bool kVerdictStraddleFilter = true;
//! A sparse or straddling visit completes its segment's verdict slot once
static constexpr bool kVerdictSlotCompletion = true;
//! A segment whose dictionary holds no match for a `contains` / `LIKE` needle is skipped
static constexpr bool kDictionarySegmentSkip = true;
//! The segment skip completes its verdict slot on the first ask
static constexpr bool kDictionarySegmentSkipEager = true;
//! The segment skip reuses its slot summary across scans
static constexpr bool kDictionarySegmentSkipReuse = true;
//! The dictionary filter compares codes with the one failing or passing code
static constexpr bool kSingleCodeFilterFastPath = true;
//! The single-code path unpacks a block of 32 codes to a survivor mask
static constexpr bool kSingleCodeFilterBlockMask = true;
//! The single-code path skips a block of 32 copies of the failing code
static constexpr bool kSingleCodeFilterBlockSkip = true;
//! The single-code path also runs under a domain verdict
static constexpr bool kSingleCodeFilterWithVerdicts = true;
//! A sparse scan of a dictionary-compressed segment publishes its decoded dictionary
static constexpr bool kPublishDictionaryOnSparseVisit = true;
//! A dictionary-compressed segment keeps references to its object-cache entries
static constexpr bool kSegmentObjectCacheReferences = true;
//! The Advanced SIMD `contains` kernel (compiled on aarch64 only)
static constexpr bool kContainsNeonKernel = true;
//! The RLE select walks the runs and the selection once
static constexpr bool kRleMergedSelect = true;

//===--------------------------------------------------------------------===//
// Scans, planning and scheduling
//===--------------------------------------------------------------------===//
//! A parallel scan splits its first large row groups into pieces
static constexpr bool kRowGroupPieces = true;
//! A table scan's pushed-filter order is shared by the threads of one scan execution: the first thread to finish an
//! observe cycle publishes its order, the others adopt it, and every filtered vector is timed
static constexpr bool kSharedFilterOrder = true;
//! A selective filter over row ids fetches its rows by row id
static constexpr bool kRowIdFetchScan = true;
//! The late-materialized row-id fetch: the index scan's task cap and split batches, the RLE fetch's block skip and
//! the buffer allocator's lazy reservation text
static constexpr bool kLateMaterializedRowIdFetch = true;
//! A constant group key beside a retained key is removed in the early duplicate-groups pass and restored above the
//! aggregate, before compressed materialization could turn it into a per-row key
static constexpr bool kRemoveConstantGroupKeys = true;
//! The count-first top-k rewrite of `ORDER BY count(*) DESC LIMIT k` aggregates
static constexpr bool kCountFirstTopK = true;
//! The count-first rewrite's smallest admitted input row count (2^20, the fused aggregate's floor)
static constexpr int64_t kCountFirstMinRows = 1048576;
//! The count-first rewrite's largest admitted input rows per distinct value of its near-unique group key
static constexpr int64_t kCountFirstMaxRowsPerKeyValue = 2;
//! The Top-N bound is applied row by row by an adaptive filter
static constexpr bool kTopNRowwiseBound = true;
//! The Top-N bound publishes an immutable copy of every value it is set to, so the scans read it without its lock
static constexpr bool kTopNBoundLockFree = true;
//! A pipeline with one task runs it inline on the scheduling thread
static constexpr bool kInlineSingleTaskPipelines = true;

//! A global-dictionary build whose statistics estimate cannot fit the global-dictionary budget beside the
//! reservations already taken is not started
static constexpr bool kGlobalDictionaryEstimateGate = true;

//! A column's global-dictionary build is spread over the scheduler's threads
static constexpr bool kGlobalDictionaryParallelBuild = true;

//! The DICT_FSST checkpoint writer appends a row group's vectors whole, so a segment ends on a vector boundary (a
//! vector that does not fit an empty segment is written row by row); a segment ended short of its block is written as a
//! whole block, and a scan reads the vector that starts where a segment ends from the next segment
static constexpr bool kVectorAlignedDictionarySegments = true;

//! A new database file at the latest storage version stores its blocks compressed, under the block-compressed
//! storage version number; off, it is created at the requested version with the fixed block layout (block-compressed
//! files are still read)
static constexpr bool kBlockCompression = true;

//! A new block-compressed file is created with split dictionaries (storage version 0x40000002, which an older reader
//! refuses): a DICT_FSST segment of a VARCHAR column keeps its local dictionary in the segment and writes its local
//! codes into a code block of its own, one per row group and column, so a scan that reads only codes never reads the
//! dictionary bytes
static constexpr bool kSplitDictionarySegments = true;
//! A DICT_FSST segment whose dictionary region reaches the split threshold (a quarter block) has its local codes split
//! into the code block, so the append's room check counts its region against the block and its codes against a
//! block of their own; off, the codes are counted against the segment's block, as if they stayed there
static constexpr bool kSplitCodesOutsideSegmentBlock = true;
//! At a table's checkpoint in such a file, each admitted VARCHAR column stores, per DICT_FSST segment, the translation
//! of its local codes into one column-wide code space numbered by first occurrence (segment order, then local order);
//! the strings stay where they first occur. A scan whose consumers read only codes (code keys, the DISTINCT bitmap, an
//! empty-string or NULL test) reads code blocks and translations and builds nothing. A smaller column builds its own
//! column-wide dictionary on first scan; the rule admits an uncollated VARCHAR column of at least this many rows ...
static constexpr bool kPersistedTranslations = true;
static constexpr int64_t kPersistedTranslationMinRows = 1048576;
//! ... whose distinct strings are at most rows / k, k = 2 (beyond that, codes gain little), and at least ...
static constexpr double kPersistedTranslationMaxDistinctShare = 0.5;
static constexpr int64_t kPersistedTranslationMinDistinct = 65536;
//! Code keys over stored translations save work per scanned row (an integer key in place of a string's hash and
//! comparison) at a setup cost per statement: the translations' directory is read, and each output string is decoded
//! from the segment where it first occurs, whose dictionary block the scan need not have read (about one block per
//! output row). The saving repays that cost only on large scans, from about 10^6 scanned rows; a scan the planner
//! estimates below this many rows after its filters groups the strings instead, as without stored translations
static constexpr int64_t kStoredCodeKeysMinScanRows = 1000000;
//! A VARCHAR column that a table scan reads for its pushed filter only (the scan does not emit it), where that filter is
//! decided on codes (an empty-string or NULL test), is read codes only through its stored translations: its dictionary
//! blocks are not read. As for code keys, the scan must be estimated at kStoredCodeKeysMinScanRows rows or more after its
//! filters: the translations' directory is read once per statement, which a small scan does not repay
static constexpr bool kFilterOnlyCodesOnly = true;
//! Under the string-predicate build guard (kStringPredicateBuildGuard) a group key or DISTINCT argument whose column has
//! stored translations is still marked, never built: its consumer reads codes only through the translations, as without
//! the string predicate. The guard's reason is the build, which reads the whole column; stored translations build nothing
static constexpr bool kStringPredicateStoredCodeKeys = true;
//! GROUP BY ... LIMIT k rewritten into a first-keys pass (FirstKeysAggregate): a VARCHAR key whose column has stored
//! translations and whose every consumer reads it for its identity only (the key itself, never its bytes) is keyed on
//! its codes in both scans, the SEMI join and the outer aggregate, and its at most k output strings are decoded after
//! the LIMIT: the scans read the column's code blocks only. As for code keys, the scan must be estimated at
//! kStoredCodeKeysMinScanRows rows or more
static constexpr bool kFirstKeysCodeKeys = true;
//! The pushed filter of a column read codes only for a key (an empty-string or NULL test) is decided once per segment
//! from the segment's translation - NULL, the empty string, any other - and the codes of the surviving rows only are
//! translated: the rows the filter drops are never translated, and no per-row filter runs over the translated codes
static constexpr bool kCodesOnlyKeyFilterPerSegment = true;
//===--------------------------------------------------------------------===//
// Regular expressions, bit-packing and the Top-N row-group order
//===--------------------------------------------------------------------===//
//! A regular-expression function's constant pattern, compiled once per thread, finds submatches with RE2's tagged DFA
//! (third_party/re2/re2/tdfa.cc) where RE2 admits the pattern; RE2's other engines run otherwise, with the same results
static constexpr bool kRegexpTaggedDFA = true;
//! regexp_replace without the 'g' option matches the input in place and writes the result string once
static constexpr bool kRegexpReplaceInPlace = true;
//! In the block-compressed file, a bit-packed group whose offsets from its minimum share a common divisor greater than
//! one stores the offsets divided by it, at the narrower width (FOR_SCALED); a scan multiplies them back
static constexpr bool kScaledFrameOfReference = true;
//! A Top-N scan whose order column is filtered by `<> ''` orders its row groups by their minimum non-empty value, so
//! the row groups holding the smallest non-empty values are scanned first and the Top-N bound tightens early
static constexpr bool kNonEmptyMinRowGroupOrder = true;

} // namespace duckdb
