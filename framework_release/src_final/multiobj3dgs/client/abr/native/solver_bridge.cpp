// Production ABI only. The packaged solver is an unchanged copy of the validated
// Proposed DP recurrence, discretization, tie-breaking and backtracking.
#include "common.hpp"

#include <cstdio>

extern "C" {
struct AbrDPOutput {
    char status[32];
    char detail[512];
    std::int64_t cost_bps;
    double utility;
    std::uint64_t visited;
    std::uint64_t budget_states;
    double elapsed_ms;
};

int abr_proposed_dp(std::size_t n, const std::int64_t* costs,
                    const double* utilities, std::int64_t budget,
                    std::int64_t delta, double timeout_seconds,
                    std::uint64_t max_dp_cells, int* selection,
                    AbrDPOutput* output) noexcept {
    if (!output) return 1;
    *output = {};
    try {
        if (!n || !costs || !utilities || !selection)
            throw std::invalid_argument("empty/null solver input");
        abrbench::Instance input;
        input.budget = budget;
        input.costs.resize(n);
        input.utilities.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (int k = 0; k < abrbench::K; ++k) {
                input.costs[i][k] = costs[i * abrbench::K + k];
                input.utilities[i][k] = utilities[i * abrbench::K + k];
            }
            selection[i] = -1;
        }
        abrbench::Options options;
        options.delta = delta;
        options.timeout_seconds = timeout_seconds;
        options.max_dp_cells = max_dp_cells;
        const auto start = std::chrono::steady_clock::now();
        const auto result = abrbench::proposed_dp(input, options);
        output->elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::snprintf(output->status, sizeof output->status, "%s", result.status.c_str());
        std::snprintf(output->detail, sizeof output->detail, "%s", result.detail.c_str());
        output->cost_bps = result.cost;
        output->utility = result.utility;
        output->visited = result.visited;
        output->budget_states = result.budget_states;
        std::copy(result.selection.begin(), result.selection.end(), selection);
        return 0;
    } catch (const std::exception& error) {
        std::snprintf(output->status, sizeof output->status, "%s", "error");
        std::snprintf(output->detail, sizeof output->detail, "%s", error.what());
        return 1;
    } catch (...) {
        std::snprintf(output->status, sizeof output->status, "%s", "error");
        std::snprintf(output->detail, sizeof output->detail, "%s", "unknown native solver exception");
        return 1;
    }
}
}
