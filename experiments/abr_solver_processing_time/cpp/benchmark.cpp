#include "common.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
using namespace abrbench;
using Solver=Result(*)(const Instance&,const Options&);
struct Job {std::string name; Solver solver; Options opts; bool stopped=false;};
struct Sample {std::string id,name,phase; int rep; Rate delta; double ns; Result result;};
std::string quote(const std::string& s) {
 std::string v="\""; for(char c:s) {if(c=='"'||c=='\\') {v+='\\'; v+=c;} else if(c=='\n') v+="\\n"; else if(c=='\r') v+="\\r"; else v+=c;} return v+'"';
}
int main(int argc,char** argv) {try {
 if(argc!=11) {std::cerr<<"usage: benchmark INPUT OUTPUT WARMUPS REPS TIMEOUT MAX_COMBINATIONS MAX_LABELS MAX_DP_CELLS DELTAS SEED\n";return 2;}
 const int warmups=std::stoi(argv[3]), reps=std::stoi(argv[4]);
 if(warmups<1||reps<1) throw std::invalid_argument("warmups and repetitions must be positive");
 Options opts; opts.timeout_seconds=std::stod(argv[5]); opts.max_combinations=std::stoull(argv[6]); opts.max_labels=std::stoull(argv[7]); opts.max_dp_cells=std::stoull(argv[8]);
 std::vector<Rate> deltas;std::stringstream ds(argv[9]);std::string part;while(std::getline(ds,part,',')) {auto d=std::stoll(part);if(d<=0) throw std::invalid_argument("delta"); deltas.push_back(d);}
 std::ifstream in(argv[1]);if(!in) throw std::runtime_error("input open failed");
 std::size_t count=0;if(!(in>>count)||!count)throw std::runtime_error("empty or invalid case count");std::vector<Instance> instances(count);
 for(auto& a:instances) {std::size_t n;in>>a.id>>n>>a.budget;a.costs.resize(n);a.utilities.resize(n);for(std::size_t i=0;i<n;++i) {for(auto& c:a.costs[i]) in>>c;for(auto& u:a.utilities[i]) in>>u;}if(!in)throw std::runtime_error("input parse failed");validate(a);}
 // All file parsing and benchmark job construction happen before the clocked solve calls.
 std::vector<Sample> samples; samples.reserve(count*(3+deltas.size())*(warmups+reps));
 std::mt19937_64 rng(std::stoull(argv[10]));
 for(const auto& a:instances) {
  std::vector<Job> jobs={{"Brute-force",brute_force,opts},{"NAVA",nava,opts},{"LTS-inspired",lts_inspired,opts}};
  for(auto d:deltas){Options o=opts;o.delta=d;jobs.push_back({"Proposed-DP",proposed_dp,o});}
  for(int round=-warmups;round<reps;++round) {
   std::vector<std::size_t> order(jobs.size());for(std::size_t j=0;j<jobs.size();++j)order[j]=j;std::shuffle(order.begin(),order.end(),rng);
   for(auto j:order){auto& job=jobs[j];if(job.stopped)continue;
    const auto start=std::chrono::steady_clock::now();
    Result result=job.solver(a,job.opts);
    const auto end=std::chrono::steady_clock::now();
    const double ns=std::chrono::duration<double,std::nano>(end-start).count();
    // Validation, retention, formatting and all I/O are outside the timed interval.
    if(!result.selection.empty()){Result verified=result;evaluate(a,verified);if(verified.cost!=result.cost||std::abs(verified.utility-result.utility)>1e-8) throw std::logic_error("result mismatch");}
    bool stop=result.status=="timeout"||result.status=="skipped";
    samples.push_back({a.id,job.name,round<0?"warmup":"measured",round<0?round+warmups:round,job.name=="Proposed-DP"?job.opts.delta:0,ns,std::move(result)});
    if(stop)job.stopped=true;
   }
  }
 }
 std::ofstream out(argv[2]);if(!out)throw std::runtime_error("output open failed");out<<std::setprecision(17);
 for(const auto& s:samples){const auto& r=s.result;out<<"{\"case_id\":"<<quote(s.id)<<",\"algorithm\":"<<quote(s.name)<<",\"phase\":"<<quote(s.phase)<<",\"repetition\":"<<s.rep<<",\"delta_bps\":"<<s.delta<<",\"elapsed_ns\":"<<s.ns<<",\"status\":"<<quote(r.status)<<",\"cost_bps\":"<<r.cost<<",\"utility\":";if(std::isfinite(r.utility))out<<r.utility;else out<<"null";
 out<<",\"selection\":[";for(std::size_t i=0;i<r.selection.size();++i){if(i)out<<',';out<<r.selection[i];}out<<"],\"visited\":"<<r.visited<<",\"pruned_dominance\":"<<r.pruned_dominance<<",\"pruned_feasibility\":"<<r.pruned_feasibility<<",\"pruned_bounds\":"<<r.pruned_bounds<<",\"budget_states\":"<<r.budget_states<<",\"solver_peak_memory_bytes\":null,\"detail\":"<<quote(r.detail)<<"}\n";}
 std::cout<<"Loaded "<<count<<" cases; wrote "<<samples.size()<<" samples.\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
