#include "common.hpp"
#include <functional>
#include <map>

namespace abrbench {
namespace {
using Utility = long double;
using Frontier = std::map<Rate, Utility>;
}

// NAVA Section III-G1 / Algorithm 1: a layered graph and recursive pulse(),
// with dominance, feasibility, and admissible objective-bound pruning.
Result nava(const Instance& a, const Options& options) {
    const auto started = std::chrono::steady_clock::now();
    validate(a);
    Result result;
    const std::size_t n = a.costs.size();
    std::vector<Rate> minimum_suffix(n + 1, 0);
    std::vector<bool> suffix_too_expensive(n + 1, false);
    std::vector<Utility> maximum_suffix(n + 1, 0);
    // Bounds are generated per invocation, hence included in measured time.
    for (std::size_t i = n; i-- > 0;) {
        const Rate minimum = *std::min_element(a.costs[i].begin(), a.costs[i].end());
        suffix_too_expensive[i] = suffix_too_expensive[i + 1] || minimum > a.budget - minimum_suffix[i + 1];
        if (!suffix_too_expensive[i]) minimum_suffix[i] = minimum + minimum_suffix[i + 1];
        const Utility maximum = *std::max_element(a.utilities[i].begin(), a.utilities[i].end());
        // Round the bound upward: it must never exclude a better complete path.
        maximum_suffix[i] = std::nextafter(maximum + maximum_suffix[i + 1],
                                         std::numeric_limits<Utility>::infinity());
    }
    // Each of the K nodes at a layer has its own nondominated prefix labels.
    // A label (r,u) dominates (r',u') iff r <= r' and u >= u'.
    std::vector<std::array<Frontier, K>> labels(n);
    std::vector<int> path(n, 0);
    Utility incumbent = -std::numeric_limits<Utility>::infinity();
    std::uint64_t live_labels = 0;
    bool stopped = false;
    std::string stop_reason;

    auto checkDominance = [&](std::size_t layer, int representation, Rate cost, Utility utility) {
        Frontier& frontier = labels[layer][representation];
        auto right = frontier.upper_bound(cost);
        if (right != frontier.begin() && std::prev(right)->second >= utility) return true;
        auto erase_at = frontier.lower_bound(cost);
        while (erase_at != frontier.end() && erase_at->second <= utility) {
            erase_at = frontier.erase(erase_at);
            --live_labels;
        }
        if (live_labels >= options.max_labels) {
            stopped = true;
            stop_reason = "configured live Pareto-label cap reached";
            return true;
        }
        frontier.emplace(cost, utility);
        ++live_labels;
        return false;
    };
    auto checkFeasibility = [&](std::size_t remaining, Rate cost) {
        return !suffix_too_expensive[remaining] && minimum_suffix[remaining] <= a.budget - cost;
    };
    auto checkBounds = [&](std::size_t remaining, Utility utility) {
        if (!std::isfinite(incumbent)) return true;
        const Utility upper = std::nextafter(utility + maximum_suffix[remaining],
                                            std::numeric_limits<Utility>::infinity());
        // A conservative arithmetic guard avoids numerical over-pruning errors.
        const Utility guard = 64 * std::numeric_limits<Utility>::epsilon() *
            (n + 1) * (1 + std::abs(upper) + std::abs(incumbent));
        return upper >= incumbent - guard;
    };

    std::function<void(std::size_t, int, Rate, Utility)> pulse;
    pulse = [&](std::size_t layer, int representation, Rate previous_cost, Utility previous_utility) {
        if (stopped) return;
        if ((result.visited & 255U) == 0U && options.timeout_seconds > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= options.timeout_seconds) {
            stopped = true;
            stop_reason = "wall-clock limit reached";
            return;
        }
        ++result.visited;
        const Rate edge_cost = a.costs[layer][representation];
        if (edge_cost > a.budget - previous_cost) {
            ++result.pruned_feasibility;
            return;
        }
        const Rate cost = previous_cost + edge_cost;
        const Utility utility = previous_utility + static_cast<Utility>(a.utilities[layer][representation]);
        if (checkDominance(layer, representation, cost, utility)) {
            if (!stopped) ++result.pruned_dominance;
            return;
        }
        if (!checkFeasibility(layer + 1, cost)) {
            ++result.pruned_feasibility;
            return;
        }
        if (!checkBounds(layer + 1, utility)) {
            ++result.pruned_bounds;
            return;
        }
        path[layer] = representation;
        if (layer + 1 == n) {
            if (utility > incumbent) {
                incumbent = utility;
                result.selection = path;
            }
            return;
        }
        for (int k = 0; k < K; ++k) {
            pulse(layer + 1, k, cost, utility);
            if (stopped) return;
        }
    };
    // The graph has a zero-cost source connected to every first-layer node.
    for (int k = 0; k < K; ++k) {
        pulse(0, k, 0, 0);
        if (stopped) break;
    }
    result.status = stopped ? "timeout" : (result.selection.empty() ? "infeasible" : "optimal");
    if (stopped) result.detail = stop_reason + "; any selection is an unproven incumbent";
    if (!result.selection.empty()) evaluate(a, result);
    return result;
}
} // namespace abrbench
