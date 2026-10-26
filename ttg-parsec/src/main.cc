#include "integrate.h"
#include "integrate_ttg.h"
#include "io.h"
#include "distributed_tree.h"

#include <ttg.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string input;
  std::string vtk_prefix;
  bool write_vtk = false;
  int steps = 10;
  double dt = 1e-3;
  double theta = 0.5;
  double eps = 0.0;
  double G = 1.0;
  unsigned max_leaf = 8;
  // Particles per coarse-grained task (grouped_ttg.h); 0 = one task per
  // (target leaf, source node) pair (bh_pass.h).
  std::uint32_t group_size = 256;
  // Particles sharing one walk of the grouped pass (grouped_ttg.h): the
  // topmost nodes with at most this many particles, or leaves.
  std::uint32_t target_size = 32;
  bool use_quadrupole = true;
  int energy_every = 1;
  bool time_steps = false;
  // Only used when --input is not given (random-particle generation).
  std::size_t n = 1000;
  unsigned seed = 1;
  double spread = 5.0;
  double vel_spread = 0.0;
};

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--input") {
      opt.input = next();
    } else if (a == "--steps") {
      opt.steps = std::stoi(next());
    } else if (a == "--dt") {
      opt.dt = std::stod(next());
    } else if (a == "--theta") {
      opt.theta = std::stod(next());
    } else if (a == "--eps") {
      opt.eps = std::stod(next());
    } else if (a == "--G") {
      opt.G = std::stod(next());
    } else if (a == "--max-leaf") {
      opt.max_leaf = static_cast<unsigned>(std::stoul(next()));
    } else if (a == "--target-size") {
      opt.target_size = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (a == "--group-size") {
      opt.group_size = static_cast<std::uint32_t>(std::stoul(next()));
    } else if (a == "--no-quadrupole") {
      opt.use_quadrupole = false;
    } else if (a == "--vtk-prefix") {
      opt.vtk_prefix = next();
      opt.write_vtk = true;
    } else if (a == "--energy-every") {
      opt.energy_every = std::stoi(next());
    } else if (a == "--time") {
      opt.time_steps = true;
    } else if (a == "--n") {
      opt.n = std::stoul(next());
    } else if (a == "--seed") {
      opt.seed = static_cast<unsigned>(std::stoul(next()));
    } else if (a == "--spread") {
      opt.spread = std::stod(next());
    } else if (a == "--vel-spread") {
      opt.vel_spread = std::stod(next());
    } else {
      throw std::runtime_error("unknown argument: " + a);
    }
  }
  return opt;
}

// Uniform-in-cube random particles (same convention as tests/bench_tree.cc
// and tools/gen_shared_ic.py), used when --input is not given so this
// benchmark can be run standalone without a prepared CSV.
std::vector<bh::Particle> make_random_particles(const Options& opt) {
  std::mt19937 rng(opt.seed);
  std::uniform_real_distribution<double> pos_dist(-opt.spread, opt.spread);
  std::uniform_real_distribution<double> vel_dist(-opt.vel_spread, opt.vel_spread);
  std::uniform_real_distribution<double> mass_dist(0.5, 1.5);

  std::vector<bh::Particle> particles(opt.n);
  for (std::size_t i = 0; i < opt.n; ++i) {
    particles[i].id = static_cast<bh::ParticleId>(i);
    particles[i].mass = mass_dist(rng);
    particles[i].pos = {pos_dist(rng), pos_dist(rng), pos_dist(rng)};
    particles[i].vel = {vel_dist(rng), vel_dist(rng), vel_dist(rng)};
  }
  return particles;
}

// Output (reports, VTK) comes from rank 0 only; every rank holds all
// particles after each step (see replicated.h).
void report(int step, const std::vector<bh::Particle>& particles, double G, const char* tag = "") {
  if (bh::this_rank() != 0) return;
  double ek = bh::kinetic_energy(particles);
  double ep = bh::potential_energy(particles, G);
  std::cout << "step " << step << tag << " E_K=" << ek << " E_P=" << ep << " E_T=" << (ek + ep)
             << "\n";
}

void report_timing(int step, const bh::StepTiming& t) {
  if (bh::this_rank() != 0) return;
  double total = t.tree_build_ms + t.traversal_ms + t.exchange_ms;
  std::cout << "step " << step << " timing: tree=" << t.tree_build_ms << "ms traversal=" << t.traversal_ms << "ms";
  if (bh::num_ranks() > 1) std::cout << " exchange=" << t.exchange_ms << "ms";
  std::cout << " total=" << total << "ms\n";
}

void add_timing(bh::StepTiming& total, const bh::StepTiming& t) {
  total.tree_build_ms += t.tree_build_ms;
  total.traversal_ms += t.traversal_ms;
  total.exchange_ms += t.exchange_ms;
}

}  // namespace

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  int rc = 0;

  try {
    Options opt = parse_args(argc, argv);
    std::vector<bh::Particle> particles =
        opt.input.empty() ? make_random_particles(opt) : bh::read_csv(opt.input);
    if (particles.empty()) throw std::runtime_error("input file contains no particles");
    if (opt.input.empty() && bh::this_rank() == 0) {
      std::cout << "generated " << particles.size() << " random particles (seed=" << opt.seed
                 << ", spread=" << opt.spread << ", vel_spread=" << opt.vel_spread << ")\n";
    }

    // The grouped pass on several ranks decomposes the domain: every rank
    // keeps an equal slice of the input, and the first step moves the
    // particles to their owners (distributed_tree.h). The per-node pass
    // keeps all particles on every rank (replicated.h).
    const bool distributed = opt.group_size > 0 && bh::num_ranks() > 1;
    if (distributed) {
      std::size_t n = particles.size(), r = bh::this_rank(), p = bh::num_ranks();
      std::vector<bh::Particle> slice(particles.begin() + n * r / p, particles.begin() + n * (r + 1) / p);
      particles.swap(slice);
    }

    bh::ForceParams fp;
    fp.theta = opt.theta;
    fp.eps2 = opt.eps * opt.eps;
    fp.G = opt.G;
    fp.use_quadrupole = opt.use_quadrupole;

    ttg::execute();

    // All particles, for output on rank 0 (collective when distributed).
    std::vector<bh::Particle> gathered;
    auto for_output = [&](const std::vector<bh::Particle>& mine) -> const std::vector<bh::Particle>& {
      if (!distributed) return mine;
      gathered = bh::gather_particles(mine);
      return gathered;
    };

    // Optimized leapfrog: rather than kick(dt/2) -> drift(dt) -> kick(dt/2)
    // as three separate host-side passes per step, each force evaluation's
    // kick (and, except for the very last, drift) is fused directly into
    // the downward pass's Sink callback (see
    // integrate_ttg.h::step_particles) and fires per-particle as soon as
    // that particle's acceleration is known - instead of materializing
    // every particle's acceleration into a vector and looping over it in a
    // separate host-side pass afterward.
    //
    // This reuses each force evaluation for both the trailing half-kick of
    // the previous step and the leading half-kick of the next (they use
    // the same acceleration, so they merge into one full-dt kick), which
    // is why the sequence below is "prime with a half-kick+drift, then
    // steps-1 full-kick+drift evaluations, then one closing half-kick with
    // no drift" - opt.steps+1 force evaluations total for opt.steps
    // advances, matching the original three-pass scheme's count exactly.
    // The one thing this can't chain through an edge: the tree rebuild
    // between evaluations is inherently sequential over the whole particle
    // set, so it stays a plain host-side barrier between passes.
    if (opt.steps > 0) {
      bh::StepTiming timing;
      bh::StepTiming total_timing;
      auto* timing_ptr = opt.time_steps ? &timing : nullptr;
      bh::GroupedWorkspace workspace;  // reused by every step (grouped pass only)
      auto run_start = std::chrono::steady_clock::now();

      particles = bh::step_particles(std::move(particles), fp, opt.max_leaf, opt.group_size, opt.target_size,
                                      opt.dt / 2.0, opt.dt,
                                      timing_ptr, &workspace);
      if (opt.energy_every > 0) report(0, for_output(particles), opt.G);
      if (opt.time_steps) {
        report_timing(0, timing);
        add_timing(total_timing, timing);
      }
      if (opt.write_vtk) {
        const auto& all = for_output(particles);
        if (bh::this_rank() == 0) bh::write_vtk(opt.vtk_prefix + "_0.vtk", all);
      }

      for (int step = 1; step < opt.steps; ++step) {
        particles =
            bh::step_particles(std::move(particles), fp, opt.max_leaf, opt.group_size, opt.target_size, opt.dt,
                                opt.dt, timing_ptr, &workspace);
        if (opt.energy_every > 0 && (step % opt.energy_every) == 0) report(step, for_output(particles), opt.G);
        if (opt.time_steps) {
          report_timing(step, timing);
          add_timing(total_timing, timing);
        }
        if (opt.write_vtk) {
          const auto& all = for_output(particles);
          if (bh::this_rank() == 0) bh::write_vtk(opt.vtk_prefix + "_" + std::to_string(step) + ".vtk", all);
        }
      }

      // Velocity has been at a half-step offset from position since the
      // prime call (standard leapfrog monitoring convention); this final
      // half-kick (no drift) fully synchronizes it at the last position
      // reached above, both for a correct final state and to show that
      // synced-vs-half-step energy agree to within the expected O(dt)
      // leapfrog discretization error.
      particles =
          bh::step_particles(std::move(particles), fp, opt.max_leaf, opt.group_size, opt.target_size,
                              opt.dt / 2.0, 0.0, timing_ptr, &workspace);
      if (opt.energy_every > 0) report(opt.steps - 1, for_output(particles), opt.G, " (synced)");
      if (opt.time_steps) {
        add_timing(total_timing, timing);
      }

      auto run_end = std::chrono::steady_clock::now();
      if (opt.time_steps && bh::this_rank() == 0) {
        double wall_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
        int n_evals = opt.steps + 1;  // matches step_particles() call count above
        double sum_ms = total_timing.tree_build_ms + total_timing.traversal_ms + total_timing.exchange_ms;
        std::cout << "timing summary: evals=" << n_evals << " wall=" << wall_ms << "ms"
                   << " tree=" << total_timing.tree_build_ms
                   << "ms (avg " << total_timing.tree_build_ms / n_evals << "ms)"
                   << " traversal=" << total_timing.traversal_ms << "ms (avg "
                   << total_timing.traversal_ms / n_evals << "ms)";
        if (bh::num_ranks() > 1) {
          std::cout << " exchange=" << total_timing.exchange_ms << "ms (avg " << total_timing.exchange_ms / n_evals
                    << "ms)";
        }
        std::cout << " sum=" << sum_ms << "ms\n";
      }
    }

    ttg::fence();
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    rc = 1;
  }

  ttg::finalize();
  return rc;
}
