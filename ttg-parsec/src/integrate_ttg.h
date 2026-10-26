#pragma once

#include "bh_pass.h"
#include "distributed_tree.h"
#include "flat_build_ttg.h"
#include "grouped_ttg.h"
#include "integrate.h"
#include "replicated.h"
#include "tree.h"
#include "types.h"

#include <chrono>
#include <stdexcept>
#include <vector>

namespace bh {

// Wall-clock breakdown of one step_particles() call, in milliseconds.
// Populated only when a non-null StepTiming* is passed in. The upward and
// downward passes no longer have a separate timing split: they're wired
// together via an edge (see bh_pass.h) rather than the caller fencing
// between them, so there's no synchronization point left to measure a
// boundary at - traversal_ms covers both.
struct StepTiming {
  double tree_build_ms = 0.0;
  double traversal_ms = 0.0;
  double exchange_ms = 0.0;  // sharing updated particles between ranks (replicated.h)
};

// Builds a fresh octree from `particles`, runs the fused upward
// (multipole) + downward (MAC-driven force) pass, and integrates each
// particle's velocity (by `kick_dt`) and, if `drift_dt != 0`, its position
// (by `drift_dt`) - AS PART OF the downward pass's per-leaf Sink callback,
// the instant that leaf's particles' accelerations are known, rather than
// materializing every particle's acceleration into a vector and looping
// over it in a separate host-side pass afterward. A leaf whose traversal
// finishes early starts integrating immediately, overlapping with leaves
// still deep in traversal.
//
// Per-node pass: the sink callback receives each particle by value (as it
// flowed through the graph - see DownwardGraph in traversal_ttg.h) rather
// than an index into a shared array, so results are collected into a fresh
// vector (keyed by the particle's own, tree-build-independent id). Grouped
// pass: particles are integrated in place in the tree's own array, so the
// result comes back in that array's (Morton) order, not by id; pass a
// `workspace` that outlives the call to reuse the build's storage across
// steps.
//
// group_size > 0 selects the coarse-grained pass (flat_build_ttg.h's
// parallel tree build, then grouped_ttg.h's walk, one task per subtree of at
// most group_size particles); 0 selects the per-node pass (Tree + bh_pass.h,
// one task per (target leaf, source node) pair). target_size only applies
// to the grouped pass (see grouped_ttg.h).
//
// kick_dt/drift_dt are independent (not both "the timestep") because the
// standard optimized leapfrog sequence - one half-kick+drift to prime,
// N-1 full-kick+drift steps, one closing half-kick with no drift - reuses
// each force evaluation for both the trailing half-kick of the previous
// step and the leading half-kick of the next (see main.cc).
// Storage of the grouped pass kept across steps (see flat_build_ttg.h).
struct GroupedWorkspace {
  FlatTree tree;
  BuildScratch scratch;
};

inline std::vector<Particle> step_particles(std::vector<Particle> particles, const ForceParams& fp,
                                             unsigned max_leaf, std::uint32_t group_size, std::uint32_t target_size,
                                             double kick_dt, double drift_dt, StepTiming* timing = nullptr,
                                             GroupedWorkspace* workspace = nullptr) {
  using clock = std::chrono::steady_clock;
  auto elapsed_ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto integrate = [kick_dt, drift_dt](Particle& p, const Vec3& a) {
    kick_one(p, a, kick_dt);
    if (drift_dt != 0.0) drift_one(p, drift_dt);
  };

  auto t0 = clock::now();
  clock::time_point t1;
  double exchange_ms = 0.0;
  // times one share_particles call (replicated.h) into exchange_ms
  auto timed = [&](auto&& f) {
    auto start = clock::now();
    f();
    exchange_ms += elapsed_ms(start, clock::now());
  };
  if (group_size > 0) {
    // Integrates in place in the tree's (Morton-ordered) particle array and
    // hands that back, taking `particles`' buffer as the tree's array for
    // the next step - so no step allocates or frees a particle array.
    GroupedWorkspace local;
    GroupedWorkspace& ws = workspace != nullptr ? *workspace : local;
    // With several ranks, `particles` are this rank's, and the tree is its
    // locally essential tree (domain decomposition, distributed_tree.h).
    if (num_ranks() > 1) {
      DistributedBuildTiming build_timing;
      build_distributed_tree(ws.tree, ws.scratch, particles, max_leaf, group_size, fp.theta, &build_timing);
      exchange_ms += build_timing.exchange_ms;
    } else {
      build_flat_tree(ws.tree, ws.scratch, particles, max_leaf, group_size);
    }
    t1 = clock::now();
    if (run_grouped_pass(ws.tree, fp, target_size, integrate) != 0) {
      throw std::logic_error("the locally essential tree lacks nodes the walk opened");
    }
    particles.swap(ws.tree.particles);
  } else {
    Tree tree(std::move(particles), max_leaf);
    t1 = clock::now();
    particles.assign(tree.num_particles(), Particle{});
    auto pass = make_bh_pass(tree, fp, [&particles, &integrate](const Particle& p, const Vec3& a) {
      Particle np = p;
      integrate(np, a);
      particles[np.id] = np;
    });
    run_bh_pass(tree, pass);
    if (num_ranks() > 1) {
      // this rank's particles: those of the target leaves it owned (see
      // seed_from_upward in traversal_ttg.h), placed by id on the others
      std::vector<Particle> mine;
      const int nranks = num_ranks(), me = this_rank();
      for (const auto& [key, node] : tree.nodes()) {
        if (!node.is_leaf() || owner_of(node.particle_begin, tree.num_particles(), nranks) != me) continue;
        for (std::uint32_t i = node.particle_begin; i < node.particle_begin + node.particle_count; ++i) {
          mine.push_back(particles[tree.particles()[i].id]);
        }
      }
      timed([&] {
        share_particles(mine.data(), mine.size(), 0, [&particles](std::uint64_t, const Particle* p, std::uint64_t n) {
          for (std::uint64_t i = 0; i < n; ++i) particles[p[i].id] = p[i];
        });
      });
    }
  }
  auto t2 = clock::now();

  if (timing != nullptr) {
    // exchange_ms is the part of the build (grouped pass) or of the
    // traversal (per-node pass) spent exchanging data between ranks
    double build_exchange_ms = group_size > 0 ? exchange_ms : 0.0;
    timing->tree_build_ms = elapsed_ms(t0, t1) - build_exchange_ms;
    timing->traversal_ms = elapsed_ms(t1, t2) - (exchange_ms - build_exchange_ms);
    timing->exchange_ms = exchange_ms;
  }

  return particles;
}

}  // namespace bh
