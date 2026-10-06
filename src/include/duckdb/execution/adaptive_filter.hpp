//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/adaptive_filter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/chrono.hpp"
#include "duckdb/common/random_engine.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/shared_ptr.hpp"

namespace duckdb {

struct AdaptiveFilterState {
	time_point<high_resolution_clock> start_time;
};

//! The shared filter order (kSharedFilterOrder): one per table scan execution (its global state), shared by that scan's
//! threads. The first thread to finish an observe cycle publishes its order and keeps exploring; the others adopt every
//! published order. Every filtered vector is timed, the ones the filters emptied too; without the shared order each
//! thread walks its own order and times only the vectors with survivors (the default)
struct AdaptiveFilterShared {
	//! 0 = nothing published, else PUBLISHED | the permutation, 4 bits per position
	atomic<uint64_t> order {0};
	static constexpr uint64_t PUBLISHED = uint64_t(1) << 63;
	static constexpr idx_t MAX_FILTERS = 15;
};

//! Whether table scans share their pushed-filter order across the threads of one scan (kSharedFilterOrder)
bool SharedFilterOrderEnabled();

struct SharedFilterOrderLocal;

class AdaptiveFilter {
public:
	explicit AdaptiveFilter(const Expression &expr);
	explicit AdaptiveFilter(const TableFilterSet &table_filters);

	vector<idx_t> permutation;

public:
	void AdaptRuntimeStatistics(double duration);

	AdaptiveFilterState BeginFilter() const;
	void EndFilter(AdaptiveFilterState state);

	//! The walk with the shared order's per-thread state passed in (SharedOrderAdaptiveFilter's side structure); the
	//! default BeginFilter / EndFilter / AdaptRuntimeStatistics above are unchanged and never reach that state
	AdaptiveFilterState BeginFilterShared(SharedFilterOrderLocal &local);
	void EndFilterShared(AdaptiveFilterState state, SharedFilterOrderLocal &local);
	//! Share this table filter's order with the other threads of one scan (ignored without the shared order, with the
	//! permutations disabled, or with one or more than MAX_FILTERS filters)
	void AttachShared(SharedFilterOrderLocal &local, shared_ptr<AdaptiveFilterShared> shared_p);

private:
	//! the walk (AdaptRuntimeStatistics' body); local is null on the default path
	void AdaptWalk(double duration, SharedFilterOrderLocal *local);
	uint64_t EncodeOrder() const;
	void Adopt(uint64_t word, SharedFilterOrderLocal &local);
	void Publish(SharedFilterOrderLocal &local);
	//! claim the explorer role with this thread's current order, else adopt the order already published
	void ClaimOrAdopt(SharedFilterOrderLocal &local);

private:
	bool disable_permutations = false;

	//! used for adaptive expression reordering
	idx_t iteration_count = 0;
	idx_t swap_idx = 0;
	idx_t right_random_border = 0;
	idx_t observe_interval = 0;
	idx_t execute_interval = 0;
	double runtime_sum = 0;
	double prev_mean = 0;
	bool observe = false;
	bool warmup = false;
	vector<idx_t> swap_likeliness;
	RandomEngine generator;
};

//! The shared order's per-thread state of one table filter, held beside the base AdaptiveFilter (whose layout it
//! leaves unchanged): the shared order slot (null = this filter walks alone) and this thread's role in it
struct SharedFilterOrderLocal {
	shared_ptr<AdaptiveFilterShared> shared;
	bool explorer = false;
	bool follower = false;
	uint64_t adopted = 0;
};

//! The table scan's adaptive filter with the shared order: the unchanged base plus the side structure, constructed and
//! deleted (as this type) only by ScanFilterInfo
class SharedOrderAdaptiveFilter : public AdaptiveFilter {
public:
	explicit SharedOrderAdaptiveFilter(const TableFilterSet &table_filters);

	SharedFilterOrderLocal local;
};
} // namespace duckdb
