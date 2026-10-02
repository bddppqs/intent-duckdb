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
static constexpr bool kFusedIntegerAggregate = true;
//! The kernel's DISTINCT class
static constexpr bool kFusedDistinctAggregate = true;
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
//! Code keys are refused when the packed group width exceeds the fused kernel's 12 bytes
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
//! A selective filter over row ids fetches its rows by row id
static constexpr bool kRowIdFetchScan = true;
//! The late-materialized row-id fetch: the index scan's task cap and split batches, the RLE fetch's block skip and
//! the buffer allocator's lazy reservation text
static constexpr bool kLateMaterializedRowIdFetch = true;
//! The count-first top-k rewrite of `ORDER BY count(*) DESC LIMIT k` aggregates
static constexpr bool kCountFirstTopK = true;
//! The count-first rewrite's smallest admitted input row count (2^20, the fused aggregate's floor)
static constexpr int64_t kCountFirstMinRows = 1048576;
//! The count-first rewrite's largest admitted input rows per distinct value of its near-unique group key
static constexpr int64_t kCountFirstMaxRowsPerKeyValue = 2;
//! The Top-N bound is applied row by row by an adaptive filter
static constexpr bool kTopNRowwiseBound = true;
//! A pipeline with one task runs it inline on the scheduling thread
static constexpr bool kInlineSingleTaskPipelines = true;

//! A global-dictionary build whose statistics estimate cannot fit the global-dictionary budget beside the
//! reservations already taken is not started
static constexpr bool kGlobalDictionaryEstimateGate = true;

//! A column's global-dictionary build is spread over the scheduler's threads
static constexpr bool kGlobalDictionaryParallelBuild = true;

//! A new database file at the latest storage version stores its blocks compressed, under the block-compressed
//! storage version number; off, it is created at the requested version with the fixed block layout (block-compressed
//! files are still read)
static constexpr bool kBlockCompression = true;

} // namespace duckdb
