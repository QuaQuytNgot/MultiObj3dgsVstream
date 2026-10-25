#include "common.hpp"

#include <cstddef>
#include <cstdint>

namespace abrbench {

Result proposed_dp(const Instance& a, const Options& options) {
    validate(a);
    if (options.delta <= 0) {
        throw std::invalid_argument("bandwidth discretization delta must be positive");
    }

    Result result;
    const std::size_t n = a.costs.size();
    const Rate budget_units = a.budget / options.delta;
    const std::uint64_t states = static_cast<std::uint64_t>(budget_units) + 1;
    result.budget_states = states;

    // Distinguish a physically infeasible decision from a feasible decision
    // excluded by conservative quantization. Avoid overflowing a sum of rates.
    Rate remaining_actual = a.budget;
    for (const auto& costs : a.costs) {
        const Rate minimum = *std::min_element(costs.begin(), costs.end());
        if (minimum > remaining_actual) {
            result.detail = "minimum actual rate exceeds the bandwidth budget";
            return result;
        }
        remaining_actual -= minimum;
    }

    std::vector<std::array<Rate, K>> quantized_costs(n);
    Rate remaining_units = budget_units;
    for (std::size_t i = 0; i < n; ++i) {
        for (int k = 0; k < K; ++k) {
            const Rate rate = a.costs[i][k];
            // Integer ceil avoids both floating point rounding and rate+delta-1
            // overflow. It cannot admit a rate above the original budget.
            quantized_costs[i][k] = rate / options.delta + (rate % options.delta != 0);
        }
        const Rate minimum = *std::min_element(quantized_costs[i].begin(), quantized_costs[i].end());
        if (minimum > remaining_units) {
            result.status = "discretized_infeasible";
            result.detail = "actual problem is feasible; ceil costs exceed the floor budget after discretization";
            return result;
        }
        remaining_units -= minimum;
    }

    // max_dp_cells bounds the N*(B+1) backtracking table. The two utility
    // rows add 16*(B+1) bytes; the parent entries use one byte each.
    if (states > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
        states > options.max_dp_cells || n > options.max_dp_cells / states) {
        result.status = "skipped";
        result.detail = "N*(B+1) exceeds max_dp_cells";
        return result;
    }
    const std::size_t width = static_cast<std::size_t>(states);
    const double unreachable = -std::numeric_limits<double>::infinity();
    constexpr std::uint8_t no_parent = 255;
    std::vector<double> previous(width, 0.0);  // DP[0][b] = 0 for all b.
    std::vector<double> current(width, unreachable);
    std::vector<std::uint8_t> parent(n * width, no_parent);
    const auto started = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < n; ++i) {
        std::fill(current.begin(), current.end(), unreachable);
        for (std::size_t b = 0; b < width; ++b) {
            if ((b & 16383U) == 0 && options.timeout_seconds > 0 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() > options.timeout_seconds) {
                result.status = "timeout";
                result.detail = "DP solve deadline exceeded";
                return result;
            }
            double best = unreachable;
            std::uint8_t choice = no_parent;
            for (int k = 0; k < K; ++k) {
                const Rate cost = quantized_costs[i][k];
                if (static_cast<std::uint64_t>(cost) > b) continue;
                const double prefix = previous[b - static_cast<std::size_t>(cost)];
                if (!std::isfinite(prefix)) continue;
                ++result.visited;
                const double candidate = prefix + a.utilities[i][k];
                // Ascending representation index and strict improvement make
                // equal-utility decisions deterministic at every DP state.
                if (candidate > best) {
                    best = candidate;
                    choice = static_cast<std::uint8_t>(k);
                }
            }
            current[b] = best;
            parent[i * width + b] = choice;
        }
        previous.swap(current);
    }

    if (!std::isfinite(previous.back())) {
        result.status = "discretized_infeasible";
        result.detail = "no feasible quantized allocation";
        return result;
    }
    result.selection.resize(n);
    std::size_t b = width - 1;
    for (std::size_t i = n; i-- > 0;) {
        const std::uint8_t k = parent[i * width + b];
        if (k == no_parent) throw std::logic_error("missing DP backtracking decision");
        result.selection[i] = k;
        b -= static_cast<std::size_t>(quantized_costs[i][k]);
    }
    evaluate(a, result);  // Recompute actual bitrate and verify its constraint.
    result.status = "optimal";
    result.detail = "optimal for conservatively discretized MCKP; not an original-problem exactness claim";
    return result;
}

}  // namespace abrbench
