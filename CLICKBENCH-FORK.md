# ClickBench fork of DuckDB v1.5.5

This repository is DuckDB v1.5.5 plus a small series of engine changes, published
so that the ClickBench entry `intent-gizmosql` can be built and reproduced from source. The
upstream history is unchanged; every change sits in the commits after upstream's `v1.5.5`
release commit (`d8cdaa33fda8df955cc76ef58a280f68f4cd43fa`) and is described in its commit message. This repository
carries a single tag, `v1.5.5-clickbench.1`, which marks the fork's head: the tree the entry was measured with.

## What changed

- RE2: batched processing of deterministic character-class runs in the BitState backtracker.
- CountZeros: compiler builtins instead of per-call lookup tables; HyperLogLog updates fed from
  the selection vector.
- Storage: decoded DICT_FSST dictionaries are kept in the evictable cache, with an admission
  back-off for segments whose dictionaries were evicted without reuse.
- Expressions: the result domain of a literal `regexp_replace` over a dictionary vector is
  memoized across requests.

None of the changes alters query results, the storage format, the SQL surface or the default
configuration; each is a general engine mechanism.

## How it was evaluated

The changes were developed and evaluated against the 43 ClickBench queries on the ClickBench
`hits` dataset, comparing paired builds with a semantic oracle for correctness and the
ClickBench timing recipe for performance. The reported DuckDB version string remains
`v1.5.5` so that runtime extensions resolve as for the upstream release.

## Where it is used

GizmoSQL, a server that embeds DuckDB, is built against this tree at
https://github.com/bddppqs/intent-gizmosql (tag `v1.38.0-clickbench.1`). The ClickBench entry `intent-gizmosql`
installs that release.

## License

DuckDB's MIT license applies unchanged; see `LICENSE`.
