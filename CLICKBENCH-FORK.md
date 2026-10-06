# ClickBench fork of DuckDB v1.5.5

DuckDB v1.5.5 plus a series of engine changes, published so that the ClickBench entry `intent-gizmosql` can
be built from source. Every change sits after upstream's release commit (`d8cdaa33fda8df955cc76ef58a280f68f4cd43fa`) and is summarised
in its commit message. Tags never move: `v1.5.5-clickbench.1` and `v1.5.5-intent.2` are the earlier releases,
`v1.5.5-intent.3` the tree the entry runs.

## What changed

- Kept from the earlier releases: RE2 character-class batching, CountZeros builtins and a decoded DICT_FSST dictionary
  cache; column-wide dictionaries with per-entry filter verdicts and segment skipping; the fused aggregation kernel;
  first-k `GROUP BY … LIMIT k`; count-first top-k; exact whole-table statistics; checkpoints that keep unchanged row
  groups; background writeback during a load and scan read-ahead; and the two fixes.
- Storage: new database files created at the latest storage version use storage version 0x40000002, which upstream DuckDB does not open; their blocks are zstd-compressed at level 9 with 64 or more threads and 3 otherwise, set on this workload, and the setting zstd_block_compression_level overrides it. A DICT_FSST segment keeps its codes in a block of their own, and a checkpoint stores, for
  large string columns, each segment's code translation into one column-wide numbering, so a scan that needs only codes
  (group keys, DISTINCT counts, empty-string and NULL tests) reads neither strings nor dictionaries. String statistics
  keep the smallest non-empty value; bit-packed groups whose offsets share a common divisor store them divided; DICT_FSST
  segments end on a vector boundary.
- Aggregation: the fused kernel takes up to three keys in 16 bytes and keys computed by a bounded deterministic function;
  a COUNT-only row folds into its partition's last row where the statistics show it pays; an ungrouped
  `count(DISTINCT x)` counts dictionary codes in a bitmap or integers in a set; a constant group key is removed before
  compressed materialization; `GROUP BY … LIMIT k` keys a column with stored translations on its codes.
- Scans and Top-N: the threads of one scan share their filter order; the Top-N bound is read without a lock; a `<> ''`
  filter prunes by the non-empty minimum and orders a Top-N scan's row groups by it.
- Regular expressions: submatches come from a lazily built tagged DFA where RE2 admits the pattern; `regexp_replace`
  without `'g'` matches in place and writes its result once.

No SQL syntax or function is added. Results are upstream's except the choice SQL leaves open under `GROUP BY … LIMIT k`
and the `MIN`/`MAX` fix. A server keeps the dictionaries, verdicts and stored translations it reads for its lifetime;
none holds query results.

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
them when the planner estimates it at `kStoredCodeKeysMinScanRows` rows (10^6) or more. These bounds, the build-time
flags and the per-architecture defaults are in `src/include/duckdb/common/tuning_defaults.hpp`:
`kGlobalDictionaryFusedDistinct`, `kFusedIntegerAggregateVarcharKeys` and `kBorrowedStringGroupKeys` are enabled on x86-64 and
disabled on arm64 (aarch64). The bundled jemalloc is built with `JEMALLOC_HAVE_MADVISE_HUGE`. Developed and evaluated
against the 43 ClickBench queries; the version string stays `v1.5.5` so that extensions resolve as for
upstream. GizmoSQL (https://github.com/bddppqs/intent-gizmosql, tag `v1.38.0-intent.3`) embeds this tree.

## License

DuckDB's MIT license applies unchanged; see `LICENSE`. The tagged DFA (`third_party/re2/re2/tdfa.cc`) is adapted from
RE2's NFA and keeps RE2's BSD license.
