//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/aggregate/min_max_string_arena.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

namespace duckdb {

class ArenaAllocator;

// MIN/MAX over VARCHAR (and the sort-key MIN/MAX of nested types) copy a non-inlined value into the state with new[]
// and free it with delete[]. Inside the grouped hash aggregate the state may instead take its bytes from the hash
// table's own aggregate arena, which outlives every row that can point into it (the arena that FIRST already uses):
// the table names that arena here, around its sink updates and its combines, and a state takes arena memory only
// when the allocator it is handed is exactly that arena (kMinMaxStringArena; off, new[] everywhere).
// The arena trades pool memory for speed (its bytes are charged to the buffer pool and never evicted); below
// dict_global::SMALL_MEMORY_LIMIT it starves the pool, so a table under a smaller memory_limit names no arena (`engaged`).
bool MinMaxArenaEnabled();
//! The arena the calling thread's grouped hash table currently updates or combines into (nullptr outside one)
ArenaAllocator *&MinMaxArenaSlot();

struct MinMaxArenaScope {
	MinMaxArenaScope(ArenaAllocator &allocator, bool engaged) : saved(MinMaxArenaSlot()) {
		if (engaged && MinMaxArenaEnabled()) {
			MinMaxArenaSlot() = &allocator;
		}
	}
	~MinMaxArenaScope() {
		MinMaxArenaSlot() = saved;
	}
	MinMaxArenaScope(const MinMaxArenaScope &) = delete;
	MinMaxArenaScope &operator=(const MinMaxArenaScope &) = delete;

private:
	ArenaAllocator *saved;
};

} // namespace duckdb
