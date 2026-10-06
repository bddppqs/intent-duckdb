# ClickBench fork of DuckDB v1.5.5

DuckDB v1.5.5 plus a series of engine changes, published so that the ClickBench entry `intent-gizmosql` can
be built from source. Every change sits after upstream's release commit (`d8cdaa33fda8df955cc76ef58a280f68f4cd43fa`) and is summarised
in its commit message. Tags never move: `v1.5.5-clickbench.1`, `v1.5.5-intent.2`, `v1.5.5-intent.3` and
`v1.5.5-intent.4` are the earlier releases, `v1.5.5-intent.5` the tree the entry runs.

## What changed

- Kept from the earlier releases: RE2 character-class batching, CountZeros builtins and a decoded DICT_FSST dictionary
  cache; column-wide dictionaries with per-entry filter verdicts and segment skipping; the fused aggregation kernel;
  first-k `GROUP BY … LIMIT k`; count-first top-k; exact whole-table statistics; checkpoints that keep unchanged row
  groups; background writeback during a load and scan read-ahead; the two fixes; dictionary codes stored apart from the
  strings with a column-wide numbering, non-empty minimum string statistics, FOR_SCALED bit-packing and DICT_FSST
  segments that end on a vector boundary; three keys in 16 bytes and computed keys in the fused kernel, the COUNT-only
  last-key fold, `count(DISTINCT x)` by a code bitmap or an integer set, and early removal of constant group keys; a
  shared filter order per scan, a lock-free Top-N bound and `<> ''` pruning by the non-empty minimum; a tagged DFA for
  regular-expression submatches and `regexp_replace` without `'g'` in place.
- Storage: new database files created at the latest storage version use storage version 0x40000003, which upstream DuckDB does not open (files at 0x40000001 and 0x40000002 still open); their blocks are zstd-compressed at level 9 with 64 or more threads and 3 otherwise, set on this workload, and the setting zstd_block_compression_level overrides it. At each checkpoint a table also writes a row-group index after its row-group pointers: the
  position of every row-group pointer and, column by column, every row group's column statistics as a load of the column
  computes them. The engine writes it by itself for every table, in the same database file. Nothing of it is read at
  attach; a table's first scan requests the index's metadata blocks at once, loads the remaining row-group pointers in
  parallel, and reads the statistics of a column it has not loaded from the index. Nested, geometry and variant columns
  store none.
- Top-N: with a dynamic filter, the threads' rows feed one boundary heap, whose front becomes the bound once it holds
  limit + offset rows. An ordered parallel scan that carries the bound can also hand out a short prefix of its first
  row group and then a bounded number of row groups at a time while the bound is set (the wave gate, off by default).
- Aggregation: an ungrouped `count(DISTINCT x)` over an integer column stored in runs takes one row per run into its
  set, and the source's tasks free the per-thread set lists once every set is built.

No SQL syntax or function is added. Results are upstream's except the choice SQL leaves open under `GROUP BY … LIMIT k`
and the `MIN`/`MAX` fix. A server keeps the dictionaries, verdicts, stored translations and row-group index entries it
reads for its lifetime; none holds query results.

## Tuning

Set on this workload, besides the zstd level rule above: `GATE_SHARE`
(`src/include/duckdb/storage/compression/dict_global/column_dictionary.hpp`): a scan estimated to return less than this
share of a table's rows does not build a column dictionary; `kStringPredicateBuildGuard`: nor does a scan with a
pushed-down string filter whose selectivity cannot be estimated; `kGlobalDictionaryWideKeyGuard`: a group key wider than
the fused kernel's 16 bytes as dictionary codes stays a string key. Count-first top-k needs an unfiltered scan of at
least `kCountFirstMinRows` rows (2^20, the fused aggregate's floor), a group key with at most
`kCountFirstMaxRowsPerKeyValue` (2) rows per value by its distinct-count statistic, and at least 16 bytes of payload
aggregate state. A checkpoint stores translations for an uncollated VARCHAR column of at least
`kPersistedTranslationMinRows` rows (2^20) whose distinct strings number at least `kPersistedTranslationMinDistinct`
(2^16) and at most half its rows, building them within half the memory limit (a quarter below 8 GiB); a scan keys on
them when the planner estimates it at `kStoredCodeKeysMinScanRows` rows (10^6) or more. A table's first scan loads its
remaining row-group pointers in parallel when at least `PARALLEL_ROW_GROUP_LOAD_MIN` (64) remain, in ranges of
max(4, ceil(N / 4T)) pointers for N pointers and T threads; with `kTopNWaveGate` enabled (it is off by default), an
ordered parallel scan under a Top-N bound hands out a prefix of `TOPN_WAVE_PREFIX_VECTORS` (4) vectors and then admits
at most `TOPN_WAVE_WINDOW` (16) hand-outs in flight while the bound is set (both in
`src/storage/table/row_group_collection.cpp`). The other bounds, the build-time flags
(among them `kPersistedRowGroupIndex`,
`kPersistedRowGroupStatistics`, `kParallelRowGroupLoad`, `kPersistedIndexReadAhead`, `kFusedRunFedDistinctSet` and
`kFusedSetSourceRelease`, enabled on both architectures, and `kTopNWaveGate`, disabled) and the per-architecture
defaults are in `src/include/duckdb/common/tuning_defaults.hpp`:
`kGlobalDictionaryFusedDistinct`, `kFusedIntegerAggregateVarcharKeys` and `kBorrowedStringGroupKeys` are enabled on x86-64 and
disabled on arm64 (aarch64). The bundled jemalloc is built with `JEMALLOC_HAVE_MADVISE_HUGE`. Developed and evaluated
against the 43 ClickBench queries; the version string stays `v1.5.5` so that extensions resolve as for
upstream. GizmoSQL (https://github.com/bddppqs/intent-gizmosql, tag `v1.38.0-intent.5`) embeds this tree.

## License

DuckDB's MIT license applies unchanged; see `LICENSE`. The tagged DFA (`third_party/re2/re2/tdfa.cc`) is adapted from
RE2's NFA and keeps RE2's BSD license.
