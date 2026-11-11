#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace abrbench {
constexpr int K=4;
using Rate=std::int64_t;
struct Instance {
 std::string id;
 std::vector<std::array<Rate,K>> costs;
 std::vector<std::array<double,K>> utilities;
 Rate budget=0;
};
struct Options {
 Rate delta=100000;
 double timeout_seconds=2.0;
 std::uint64_t max_combinations=1000000;
 std::uint64_t max_labels=500000;
 std::uint64_t max_dp_cells=50000000;
};
struct Result {
 std::string status="infeasible"; // optimal, infeasible, discretized_infeasible, timeout, skipped
 std::vector<int> selection;
 double utility=-std::numeric_limits<double>::infinity();
 Rate cost=0;
 std::uint64_t visited=0;
 std::uint64_t pruned_dominance=0, pruned_feasibility=0, pruned_bounds=0;
 std::uint64_t budget_states=0;
 std::string detail;
};
inline void validate(const Instance& a) {
 if(a.costs.empty() || a.costs.size()!=a.utilities.size() || a.budget<0) throw std::invalid_argument("invalid instance shape/budget");
 for(std::size_t i=0;i<a.costs.size();++i) for(int k=0;k<K;++k) if(a.costs[i][k]<0 || !std::isfinite(a.utilities[i][k])) throw std::invalid_argument("invalid cost/utility");
}
inline void evaluate(const Instance& a, Result& r) {
 if(r.selection.size()!=a.costs.size()) throw std::logic_error("selection shape");
 r.cost=0; r.utility=0;
 for(std::size_t i=0;i<r.selection.size();++i) {int k=r.selection[i]; if(k<0||k>=K) throw std::logic_error("selection index"); r.cost+=a.costs[i][k]; r.utility+=a.utilities[i][k];}
 if(r.cost>a.budget) throw std::logic_error("solver exceeded actual bandwidth budget");
}
Result brute_force(const Instance&,const Options&);
Result nava(const Instance&,const Options&);
Result lts_inspired(const Instance&,const Options&);
Result proposed_dp(const Instance&,const Options&);
}
