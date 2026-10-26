#pragma once

#include "../src/bh_pass.h"
#include "../src/flat_build_ttg.h"
#include "../src/grouped_ttg.h"
#include "../src/tree.h"

#include <ttg.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace bh_test {

// Runs the fused upward+downward pass once (per-node pass for group_size 0,
// grouped pass otherwise) and returns {particles, matching
// acceleration vector}, both indexed by each particle's own id (the sink
// delivers particles by value as they flowed through the graph, not by an
// index into tree.particles() - see traversal_ttg.h) - mirrors main.cc's
// compute_accel helper, but as a free function here since main.cc's is
// file-local.
inline std::pair<std::vector<bh::Particle>, std::vector<bh::Vec3>> run_bh(
    std::vector<bh::Particle> particles, const bh::ForceParams& fp, unsigned max_leaf,
    std::uint32_t group_size = 0, std::uint32_t target_size = 0) {
  std::vector<bh::Particle> out_particles(particles.size());
  std::vector<bh::Vec3> accel(particles.size());
  auto on_result = [&](const auto& p, const bh::Vec3& a) {  // const Particle& or Particle&
    out_particles[p.id] = p;
    accel[p.id] = a;
  };
  if (group_size > 0) {
    bh::FlatTree flat = bh::build_flat_tree(std::move(particles), max_leaf, group_size);
    bh::run_grouped_pass(flat, fp, target_size, on_result);
  } else {
    bh::Tree tree(std::move(particles), max_leaf);
    auto pass = bh::make_bh_pass(tree, fp, on_result);
    bh::run_bh_pass(tree, pass);
  }

  return {std::move(out_particles), std::move(accel)};
}

}  // namespace bh_test
