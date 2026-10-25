#include "common.hpp"

#include <functional>
#include <iostream>
#include <random>
#include <sstream>

namespace {
using namespace abrbench;
int assertions = 0;

void require(bool condition, const std::string& message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, const std::string& message) {
    const double tolerance = 1e-10 * std::max(1.0, std::abs(expected));
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected));
}

struct Oracle {
    bool feasible = false;
    double utility = -std::numeric_limits<double>::infinity();
    std::vector<int> selection;
};

// A deliberately simple independent enumerator used only in these tests. It
// materializes every K^N leaf (including infeasible leaves), then checks total
// cost. It does not call any implementation under test or prune the tree.
Oracle enumerate(const Instance& instance, Rate delta = 0) {
    Oracle answer;
    std::vector<int> selected(instance.costs.size(), 0);
    const Rate budget = delta ? instance.budget / delta : instance.budget;
    std::function<void(std::size_t)> visit = [&](std::size_t i) {
        if (i < selected.size()) {
            for (int k = 0; k < K; ++k) {
                selected[i] = k;
                visit(i + 1);
            }
            return;
        }
        Rate rate = 0;
        double utility = 0;
        for (std::size_t j = 0; j < selected.size(); ++j) {
            const Rate cost = instance.costs[j][selected[j]];
            rate += delta ? cost / delta + (cost % delta != 0) : cost;
            utility += instance.utilities[j][selected[j]];
        }
        if (rate <= budget && (!answer.feasible || utility > answer.utility)) {
            answer.feasible = true;
            answer.utility = utility;
            answer.selection = selected;
        }
    };
    visit(0);
    return answer;
}

void verify_selection(const Instance& instance, const Result& result, const std::string& label) {
    require(result.selection.size() == instance.costs.size(), label + " selection length");
    Rate sum = 0;
    double utility = 0;
    for (std::size_t i = 0; i < result.selection.size(); ++i) {
        const int k = result.selection[i];
        require(k >= 0 && k < K, label + " representation index");
        sum += instance.costs[i][k];
        utility += instance.utilities[i][k];
    }
    require(sum <= instance.budget, label + " violates original bandwidth constraint");
    require(sum == result.cost, label + " incorrect returned cost");
    close(result.utility, utility, label + " incorrect returned utility");
}

void check_instance(const Instance& instance, Rate delta) {
    const Oracle exact = enumerate(instance);
    const Oracle quantized = enumerate(instance, delta);
    Options options;
    options.delta = delta;
    options.timeout_seconds = 60;
    options.max_combinations = 10000000;
    options.max_labels = 1000000;
    options.max_dp_cells = 10000000;
    const Instance original = instance;
    const std::array<std::pair<const char*, Result (*)(const Instance&, const Options&)>, 3> solvers{{
        {"brute-force", brute_force}, {"NAVA", nava}, {"LTS-inspired", lts_inspired}
    }};
    for (const auto& entry : solvers) {
        const Result result = entry.second(instance, options);
        const std::string label = instance.id + " " + entry.first;
        require(instance.costs == original.costs && instance.utilities == original.utilities && instance.budget == original.budget,
                label + " mutated shared input");
        if (exact.feasible) {
            require(result.status == "optimal", label + " unexpected status " + result.status);
            verify_selection(instance, result, label);
            close(result.utility, exact.utility, label + " differs from independent exact oracle");
        } else {
            require(result.status == "infeasible", label + " failed to identify infeasibility");
            require(result.selection.empty(), label + " returned infeasible selection");
        }
    }
    const Result result = proposed_dp(instance, options);
    const std::string label = instance.id + " proposed-DP";
    require(instance.costs == original.costs && instance.utilities == original.utilities && instance.budget == original.budget,
            label + " mutated shared input");
    if (quantized.feasible) {
        require(result.status == "optimal", label + " unexpected status " + result.status);
        verify_selection(instance, result, label);
        close(result.utility, quantized.utility, label + " differs from quantized enumerator");
        require(exact.feasible, label + " quantization admitted infeasible original problem");
        require(result.utility <= exact.utility + 1e-10 * std::max(1.0, std::abs(exact.utility)),
                label + " negative objective gap");
        Rate quantized_sum = 0;
        for (std::size_t i = 0; i < result.selection.size(); ++i) {
            const Rate rate = instance.costs[i][result.selection[i]];
            quantized_sum += rate / delta + (rate % delta != 0);
        }
        require(quantized_sum <= instance.budget / delta, label + " violates discretized bandwidth constraint");
        if (delta == 1) close(result.utility, exact.utility, label + " unit-step exactness");
    } else {
        require(result.status == (exact.feasible ? "discretized_infeasible" : "infeasible"),
                label + " incorrect infeasibility classification: " + result.status);
        require(result.selection.empty(), label + " returned infeasible selection");
    }
}

void edge_cases() {
    Instance known{"known-optimum", {{{1, 3, 5, 7}}, {{1, 2, 4, 8}}},
                   {{{1, 5, 8, 10}}, {{2, 4, 7, 11}}}, 7};
    const auto expected = enumerate(known);
    close(expected.utility, 12, "hand-computed oracle objective");
    require(expected.selection == std::vector<int>({1, 2}), "hand-computed oracle selection");
    check_instance(known, 1);
    check_instance(known, 3);
    check_instance(Instance{"infeasible", {{{2, 3, 4, 5}}, {{3, 4, 5, 6}}},
                            {{{9, 8, 7, 6}}, {{1, 2, 3, 4}}}, 4}, 2);
    check_instance(Instance{"rounding-excludes", {{{1, 11, 12, 13}}, {{1, 11, 12, 13}}},
                            {{{1, 5, 6, 7}}, {{1, 5, 6, 7}}}, 2}, 2);
    check_instance(Instance{"ceil-boundary", {{{10, 11, 20, 21}}, {{0, 1, 10, 11}}},
                            {{{1, 8, 9, 20}}, {{0, 7, 8, 25}}}, 20}, 10);
    check_instance(Instance{"zero-budget-and-rates", {{{0, 0, 1, 4}}, {{0, 0, 0, 3}}},
                            {{{-2, -1, 50, 60}}, {{0, 0, 0, 100}}}, 0}, 100);
    const Instance tied{"ties", {{{0, 0, 0, 0}}, {{0, 0, 0, 0}}},
                        {{{3, 3, 3, 3}}, {{-1, -1, -1, -1}}}, 0};
    check_instance(tied, 1);
    const Result first = proposed_dp(tied, Options{});
    const Result second = proposed_dp(tied, Options{});
    require(first.selection == std::vector<int>({0, 0}) && first.selection == second.selection,
            "DP ties must be deterministic");
    check_instance(Instance{"negative-and-zero-weights", {{{1, 2, 3, 4}}, {{4, 3, 2, 1}}},
                            {{{0, 0, 0, 0}}, {{-80, -60, -40, -20}}}, 5}, 1);

    // Integer rounding near int64 maximum must not use rate+delta-1.
    const Rate maximum = std::numeric_limits<Rate>::max();
    Instance wide{"int64-rounding", {{{maximum, maximum - 1, 0, 1}}}, {{{10, 9, 0, 1}}}, maximum};
    Options options;
    options.delta = maximum;
    Result wide_result = proposed_dp(wide, options);
    require(wide_result.status == "optimal" && wide_result.selection == std::vector<int>({0}),
            "int64-safe ceil/floor rounding");
    verify_selection(wide, wide_result, "int64-rounding");

    options.delta = 1;
    options.max_dp_cells = 10;
    require(proposed_dp(known, options).status == "skipped", "DP resource ceiling");
    options.delta = 0;
    bool threw = false;
    try { proposed_dp(known, options); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "zero delta must be rejected");
    options.delta = -1;
    threw = false;
    try { proposed_dp(known, options); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "negative delta must be rejected");
}

void randomized_cases() {
    std::mt19937_64 random(20271025);
    constexpr int trials = 350;
    const std::array<Rate, 7> deltas{{1, 2, 3, 7, 10, 11, 20}};
    for (int trial = 0; trial < trials; ++trial) {
        const std::size_t n = 1 + random() % 7;
        Instance instance;
        instance.id = "random-" + std::to_string(trial);
        instance.costs.resize(n);
        instance.utilities.resize(n);
        Rate maximum_sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            // Include nonmonotone representations, duplicate costs, ties,
            // negative utilities, and zero utilities/visibility weights.
            const double weight = static_cast<int>(random() % 9) - 3;
            for (int k = 0; k < K; ++k) {
                instance.costs[i][k] = random() % 61;
                instance.utilities[i][k] = weight * static_cast<int>(random() % 31) / 4.0;
            }
            maximum_sum += *std::max_element(instance.costs[i].begin(), instance.costs[i].end());
        }
        instance.budget = random() % (maximum_sum + 1);
        check_instance(instance, deltas[trial % deltas.size()]);
    }
}
}  // namespace

int main() {
    try {
        edge_cases();
        randomized_cases();
        std::cout << "PASS: 350 randomized instances and edge cases; " << assertions << " assertions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
