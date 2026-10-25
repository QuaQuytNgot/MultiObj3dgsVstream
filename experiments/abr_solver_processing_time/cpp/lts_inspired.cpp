#include "common.hpp"
#include <functional>

namespace abrbench {

// Independent full-space candidate exploration after adapting LTS's decision
// units to independent object x CRF choices. No LTS layer dependencies remain.
Result lts_inspired(const Instance& a, const Options& options) {
    const auto started = std::chrono::steady_clock::now();
    validate(a);
    Result result;
    std::uint64_t combinations = 1;
    for (std::size_t i = 0; i < a.costs.size(); ++i) {
        if (combinations > options.max_combinations / K) {
            result.status = "skipped";
            result.detail = "full-space K^N exceeds configured max_combinations; no search executed";
            return result;
        }
        combinations *= K;
    }
    std::vector<int> candidate(a.costs.size(), 0);
    long double best = -std::numeric_limits<long double>::infinity();
    bool stopped = false;
    // Unlike Algorithm A's flat odometer this recursively constructs a complete
    // candidate and carries its cost/utility. It still visits every leaf, even
    // when an intermediate assignment already exceeds the bandwidth budget.
    std::function<void(std::size_t, Rate, long double, bool)> explore;
    explore = [&](std::size_t i, Rate cost, long double utility, bool feasible) {
        if (stopped) return;
        if (i == candidate.size()) {
            if ((result.visited & 1023U) == 0U && options.timeout_seconds > 0 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= options.timeout_seconds) {
                stopped = true;
                return;
            }
            ++result.visited;
            if (feasible && utility > best) {
                best = utility;
                result.selection = candidate;
            }
            return;
        }
        for (int k = 0; k < K; ++k) {
            candidate[i] = k;
            const bool next_feasible = feasible && a.costs[i][k] <= a.budget - cost;
            const Rate next_cost = next_feasible ? cost + a.costs[i][k] : cost;
            explore(i + 1, next_cost, utility + static_cast<long double>(a.utilities[i][k]), next_feasible);
            if (stopped) return;
        }
    };
    explore(0, 0, 0, true);
    result.status = stopped ? "timeout" : (result.selection.empty() ? "infeasible" : "optimal");
    if (stopped) result.detail = "wall-clock limit reached; any selection is an unproven incumbent";
    if (!result.selection.empty()) evaluate(a, result);
    return result;
}
} // namespace abrbench
