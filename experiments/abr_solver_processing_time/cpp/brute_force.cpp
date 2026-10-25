#include "common.hpp"

namespace abrbench {

// Complete Cartesian-product enumeration. No branch pruning or dynamic program.
Result brute_force(const Instance& a, const Options& options) {
    const auto started = std::chrono::steady_clock::now();
    validate(a);
    Result result;
    std::uint64_t combinations = 1;
    for (std::size_t i = 0; i < a.costs.size(); ++i) {
        if (combinations > options.max_combinations / K) {
            result.status = "skipped";
            result.detail = "K^N exceeds configured max_combinations; no search executed";
            return result;
        }
        combinations *= K;
    }
    std::vector<int> candidate(a.costs.size(), 0);
    long double best = -std::numeric_limits<long double>::infinity();
    bool timed_out = false;
    for (std::uint64_t candidate_number = 0; candidate_number < combinations; ++candidate_number) {
        if ((candidate_number & 1023U) == 0U && options.timeout_seconds > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= options.timeout_seconds) {
            timed_out = true;
            break;
        }
        Rate cost = 0;
        bool feasible = true;
        long double utility = 0;
        // Recompute every complete candidate from the same prepared input arrays.
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            const int k = candidate[i];
            utility += static_cast<long double>(a.utilities[i][k]);
            // This subtraction avoids overflowing int64 when rejecting a candidate.
            if (feasible && a.costs[i][k] <= a.budget - cost) cost += a.costs[i][k];
            else feasible = false;
        }
        ++result.visited;
        if (feasible && utility > best) {
            best = utility;
            result.selection = candidate;
        }
        // Lexicographic order makes exact utility ties deterministic.
        for (std::size_t i = candidate.size(); i-- > 0;) {
            if (++candidate[i] < K) break;
            candidate[i] = 0;
        }
    }
    result.status = timed_out ? "timeout" : (result.selection.empty() ? "infeasible" : "optimal");
    if (timed_out) result.detail = "wall-clock limit reached; any selection is an unproven incumbent";
    if (!result.selection.empty()) evaluate(a, result);
    return result;
}
} // namespace abrbench
