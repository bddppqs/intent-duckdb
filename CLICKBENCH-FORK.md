# ClickBench fork of DuckDB v1.5.5

DuckDB v1.5.5 plus a series of engine changes, published so that the ClickBench entry `intent-gizmosql` can
be built from source. Every change sits after upstream's release commit (`d8cdaa33fda8df955cc76ef58a280f68f4cd43fa`) and is summarised
in its commit message. Tags never move: `v1.5.5-clickbench.1` is the first release, `v1.5.5-intent.2` the
measured tree.

## What changed

- Kept from the first release: RE2 character-class batching, CountZeros builtins, a decoded DICT_FSST dictionary cache.
- Column-wide dictionaries for DICT_FSST string columns, built on first scan and used by filters and aggregates;
  string-filter verdicts per dictionary entry skip segments without a match.
- Aggregation and plans: a fused kernel for small fixed-width group keys; `GROUP BY … LIMIT k` without `ORDER BY` keeps the
  first k groups found; count-first top-k for large near-unique groupings.
- Storage: new database files created at the latest storage version store zstd-compressed blocks under storage version 0x40000001, which upstream DuckDB does not open; the zstd level is 9 at 64 or more threads and 3 otherwise, set on this workload, and the setting zstd_block_compression_level overrides it. Exact table statistics answer whole-table `COUNT(*)`, `MIN` and `MAX`; checkpoints no
  longer merge row groups that are all unchanged on disk; device writeback starts in the background during a load; scans
  read ahead.
- Fixes: WAL replay applies a transaction's row groups only at its commit record (a durability fix); `MIN`/`MAX` over
  only NaN or infinite values return the scan's answer.

No SQL syntax or function is added. Results are upstream's except the choice SQL leaves open under `GROUP BY … LIMIT k`
and the `MIN`/`MAX` fix. A server keeps the dictionaries and verdicts for its lifetime; none holds query results.

## Tuning

Set on this workload, besides the zstd level rule above: `GATE_SHARE`
(`src/include/duckdb/storage/compression/dict_global/column_dictionary.hpp`): a scan estimated to return less than this
share of a table's rows does not build a column dictionary; `kStringPredicateBuildGuard`: nor does a scan with a
pushed-down string filter whose selectivity cannot be estimated; `kGlobalDictionaryWideKeyGuard`: a group key wider than
the fused kernel's 12 bytes as dictionary codes stays a string key. Count-first top-k needs an unfiltered scan of at
least `kCountFirstMinRows` rows (2^20, the fused aggregate's floor), a group key with at most
`kCountFirstMaxRowsPerKeyValue` (2) rows per value by its distinct-count statistic, and at least 16 bytes of payload
aggregate state. The two guards, the two count-first bounds, the build-time flags and the per-architecture defaults are
in `src/include/duckdb/common/tuning_defaults.hpp`: `kGlobalDictionaryFusedDistinct`, `kFusedIntegerAggregateVarcharKeys` and
`kBorrowedStringGroupKeys` are enabled on x86-64 and disabled on arm64 (aarch64). The bundled jemalloc is built with
`JEMALLOC_HAVE_MADVISE_HUGE`. Developed and evaluated against the 43 ClickBench queries; the version string stays
`v1.5.5` so that extensions resolve as for upstream. GizmoSQL (https://github.com/bddppqs/intent-gizmosql, tag
`v1.38.0-intent.2`) embeds this tree.

## License

DuckDB's MIT license applies unchanged; see `LICENSE`.
