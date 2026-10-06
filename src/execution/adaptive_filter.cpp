#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/optimizer/expression_heuristics.hpp"
#include "duckdb/execution/adaptive_filter.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/vector.hpp"

#include "duckdb/common/tuning_defaults.hpp"

namespace duckdb {

bool SharedFilterOrderEnabled() {
	return kSharedFilterOrder;
}

AdaptiveFilter::AdaptiveFilter(const Expression &expr) : observe_interval(10), execute_interval(20), warmup(true) {
	auto &conj_expr = expr.Cast<BoundConjunctionExpression>();
	D_ASSERT(conj_expr.children.size() > 1);
	for (idx_t idx = 0; idx < conj_expr.children.size(); idx++) {
		permutation.push_back(idx);
		if (conj_expr.children[idx]->CanThrow()) {
			disable_permutations = true;
		}
		if (idx != conj_expr.children.size() - 1) {
			swap_likeliness.push_back(100);
		}
	}
	right_random_border = 100 * (conj_expr.children.size() - 1);
}

AdaptiveFilter::AdaptiveFilter(const TableFilterSet &table_filters)
    : observe_interval(10), execute_interval(20), warmup(true) {
	permutation = ExpressionHeuristics::GetInitialOrder(table_filters);
	for (idx_t idx = 1; idx < table_filters.filters.size(); idx++) {
		swap_likeliness.push_back(100);
	}
	right_random_border = 100 * (table_filters.filters.size() - 1);
}

AdaptiveFilterState AdaptiveFilter::BeginFilter() const {
	if (permutation.size() <= 1 || disable_permutations) {
		return AdaptiveFilterState();
	}
	AdaptiveFilterState state;
	state.start_time = high_resolution_clock::now();
	return state;
}

void AdaptiveFilter::EndFilter(AdaptiveFilterState state) {
	if (permutation.size() <= 1 || disable_permutations) {
		// nothing to permute
		return;
	}
	auto end_time = high_resolution_clock::now();
	AdaptRuntimeStatistics(duration_cast<duration<double>>(end_time - state.start_time).count());
}

SharedOrderAdaptiveFilter::SharedOrderAdaptiveFilter(const TableFilterSet &table_filters)
    : AdaptiveFilter(table_filters) {
}

void AdaptiveFilter::AttachShared(SharedFilterOrderLocal &local, shared_ptr<AdaptiveFilterShared> shared_p) {
	if (!SharedFilterOrderEnabled() || disable_permutations || permutation.size() <= 1 ||
	    permutation.size() > AdaptiveFilterShared::MAX_FILTERS) {
		return;
	}
	local.shared = std::move(shared_p);
}

uint64_t AdaptiveFilter::EncodeOrder() const {
	uint64_t word = AdaptiveFilterShared::PUBLISHED;
	for (idx_t i = 0; i < permutation.size(); i++) {
		word |= uint64_t(permutation[i]) << (4 * i);
	}
	return word;
}

void AdaptiveFilter::Adopt(uint64_t word, SharedFilterOrderLocal &local) {
	for (idx_t i = 0; i < permutation.size(); i++) {
		permutation[i] = (word >> (4 * i)) & 0xF;
	}
	local.adopted = word;
	local.follower = true;
}

void AdaptiveFilter::Publish(SharedFilterOrderLocal &local) {
	auto word = EncodeOrder();
	if (word != local.adopted) {
		local.shared->order.store(word, std::memory_order_release);
		local.adopted = word;
	}
}

void AdaptiveFilter::ClaimOrAdopt(SharedFilterOrderLocal &local) {
	uint64_t expected = 0;
	auto word = EncodeOrder();
	if (local.shared->order.compare_exchange_strong(expected, word, std::memory_order_acq_rel)) {
		local.explorer = true;
		local.adopted = word;
	} else {
		Adopt(expected, local);
	}
}

AdaptiveFilterState AdaptiveFilter::BeginFilterShared(SharedFilterOrderLocal &local) {
	if (local.shared && !local.explorer) {
		// a follower (or a thread not yet settled) runs the published order, read once per vector
		auto word = local.shared->order.load(std::memory_order_acquire);
		if (word != local.adopted) {
			Adopt(word, local);
		}
	}
	if (permutation.size() <= 1 || disable_permutations || local.follower) {
		return AdaptiveFilterState();
	}
	AdaptiveFilterState state;
	state.start_time = high_resolution_clock::now();
	return state;
}

void AdaptiveFilter::EndFilterShared(AdaptiveFilterState state, SharedFilterOrderLocal &local) {
	if (permutation.size() <= 1 || disable_permutations || local.follower) {
		// nothing to permute (a follower does not explore)
		return;
	}
	auto end_time = high_resolution_clock::now();
	AdaptWalk(duration_cast<duration<double>>(end_time - state.start_time).count(), &local);
}

void AdaptiveFilter::AdaptRuntimeStatistics(double duration) {
	AdaptWalk(duration, nullptr);
}

void AdaptiveFilter::AdaptWalk(double duration, SharedFilterOrderLocal *local) {
	iteration_count++;
	runtime_sum += duration;

	D_ASSERT(!disable_permutations);
	if (!warmup) {
		// the last swap was observed
		if (observe && iteration_count == observe_interval) {
			// keep swap if runtime decreased, else reverse swap
			if (prev_mean - (runtime_sum / static_cast<double>(iteration_count)) <= 0) {
				// reverse swap because runtime didn't decrease
				std::swap(permutation[swap_idx], permutation[swap_idx + 1]);

				// decrease swap likeliness, but make sure there is always a small likeliness left
				if (swap_likeliness[swap_idx] > 1) {
					swap_likeliness[swap_idx] /= 2;
				}
			} else {
				// keep swap because runtime decreased, reset likeliness
				swap_likeliness[swap_idx] = 100;
				if (local && local->explorer) {
					// the explorer republishes only a measured improvement
					Publish(*local);
				}
			}
			observe = false;

			// reset values
			iteration_count = 0;
			runtime_sum = 0.0;
			if (local && local->shared && !local->explorer) {
				// the first thread to finish an observe cycle publishes its order and keeps exploring
				ClaimOrAdopt(*local);
			}
		} else if (!observe && iteration_count == execute_interval) {
			// save old mean to evaluate swap
			prev_mean = runtime_sum / static_cast<double>(iteration_count);

			// get swap index and swap likeliness
			// a <= i <= b
			auto random_number = generator.NextRandomInteger(1, NumericCast<uint32_t>(right_random_border));

			swap_idx = random_number / 100;                    // index to be swapped
			idx_t likeliness = random_number - 100 * swap_idx; // random number between [0, 100)

			// check if swap is going to happen
			if (swap_likeliness[swap_idx] > likeliness) { // always true for the first swap of an index
				// swap
				std::swap(permutation[swap_idx], permutation[swap_idx + 1]);

				// observe whether swap will be applied
				observe = true;
			}

			// reset values
			iteration_count = 0;
			runtime_sum = 0.0;
		}
	} else {
		if (iteration_count == 5) {
			// initially set all values
			iteration_count = 0;
			runtime_sum = 0.0;
			observe = false;
			warmup = false;
		}
	}
}

} // namespace duckdb
