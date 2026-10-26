#include "../src/brute_force.h"
#include "test_common.h"

#include <ttg.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace bh;

namespace {

std::vector<Particle> random_particles(std::size_t n, unsigned seed, double spread,
                                        Vec3 center = {}) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> pos_dist(-spread, spread);
  std::uniform_real_distribution<double> mass_dist(0.5, 1.5);
  std::vector<Particle> particles(n);
  for (std::size_t i = 0; i < n; ++i) {
    particles[i].id = static_cast<ParticleId>(i);
    particles[i].mass = mass_dist(rng);
    particles[i].pos = {center.x + pos_dist(rng), center.y + pos_dist(rng), center.z + pos_dist(rng)};
  }
  return particles;
}

// Global-normalized error metric: sqrt(sum|a_test-a_ref|^2) / sqrt(sum|a_ref|^2).
// Avoids the blow-up a per-particle relative error would have for particles
// whose true net force happens to be near zero.
double relative_error(const std::vector<Vec3>& test, const std::vector<Vec3>& ref) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    num += norm2(test[i] - ref[i]);
    den += norm2(ref[i]);
  }
  return std::sqrt(num / den);
}

}  // namespace

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();
  int rc = 0;

  // Group sizes: 0 = per-node pass (bh_pass.h); 16 = grouped pass with many
  // groups and an aggregated top tree; 1<<20 = grouped pass where the whole
  // tree is a single group (no top_up at all).
  for (std::uint32_t gs : {0u, 16u, 1u << 20}) {
  std::cout << "--- group_size=" << gs << "\n";
  {
    // theta=0 must degenerate Barnes-Hut into brute force (whitepaper's own
    // stated property of Eq. 10), so the tree code should reproduce the
    // O(N^2) reference to floating-point summation-order tolerance.
    auto particles = random_particles(200, /*seed=*/1, /*spread=*/10.0);
    ForceParams fp;
    fp.theta = 0.0;
    fp.eps2 = 1e-6;
    fp.G = 1.0;
    fp.use_quadrupole = true;

    auto [bh_particles, bh_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/4, gs);
    auto ref_accel = brute_force_accel(bh_particles, fp.eps2, fp.G);

    double err = relative_error(bh_accel, ref_accel);
    std::cout << "theta=0 relative error: " << err << "\n";
    if (err > 1e-8) {
      std::cerr << "FAIL: theta=0 should reproduce brute force to floating-point tolerance\n";
      rc = 1;
    } else {
      std::cout << "PASS: theta=0 matches brute force\n";
    }
  }

  {
    // Two tight, well-separated clusters at moderate theta: each cluster's
    // internal spread gives it a nontrivial quadrupole moment, so adding
    // the quadrupole term should measurably reduce the force error versus
    // a monopole-only approximation - the concrete check that the
    // parallel-axis-shift/quadrupole formula's sign and factor are correct.
    auto cluster_a = random_particles(100, /*seed=*/2, /*spread=*/1.0, Vec3{-50.0, 0.0, 0.0});
    auto cluster_b = random_particles(100, /*seed=*/3, /*spread=*/1.0, Vec3{50.0, 0.0, 0.0});
    std::vector<Particle> particles = cluster_a;
    particles.insert(particles.end(), cluster_b.begin(), cluster_b.end());
    for (std::size_t i = 0; i < particles.size(); ++i) particles[i].id = static_cast<ParticleId>(i);

    ForceParams fp;
    fp.theta = 0.6;
    fp.eps2 = 1e-6;
    fp.G = 1.0;

    fp.use_quadrupole = false;
    auto [mono_particles, mono_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/4, gs);
    auto ref_accel_mono = brute_force_accel(mono_particles, fp.eps2, fp.G);
    double mono_err = relative_error(mono_accel, ref_accel_mono);

    fp.use_quadrupole = true;
    auto [quad_particles, quad_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/4, gs);
    auto ref_accel_quad = brute_force_accel(quad_particles, fp.eps2, fp.G);
    double quad_err = relative_error(quad_accel, ref_accel_quad);

    std::cout << "monopole-only relative error: " << mono_err << "\n";
    std::cout << "monopole+quadrupole relative error: " << quad_err << "\n";
    if (!(quad_err < mono_err)) {
      std::cerr << "FAIL: quadrupole should reduce force error versus monopole-only\n";
      rc = 1;
    } else {
      std::cout << "PASS: quadrupole term reduces force error versus monopole-only\n";
    }
  }
  }

  {
    // The grouped pass uses the same leaf-wide MAC as the per-node pass, so
    // at any theta both must make identical accept/expand decisions and
    // agree up to floating-point summation order and the SIMD kernel's
    // rsqrt precision (~5e-11 on NEON, see simd.h) - a single differing
    // decision shows up orders of magnitude above that.
    auto particles = random_particles(2000, /*seed=*/4, /*spread=*/10.0);
    ForceParams fp;
    fp.theta = 0.5;
    fp.eps2 = 1e-6;
    fp.G = 1.0;
    fp.use_quadrupole = true;
    auto [node_particles, node_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/8, 0);
    auto [group_particles, group_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/8, 64);
    double err = relative_error(group_accel, node_accel);
    std::cout << "grouped vs per-node relative difference at theta=0.5: " << err << "\n";
    if (err > 1e-8) {
      std::cerr << "FAIL: grouped pass should match the per-node pass\n";
      rc = 1;
    } else {
      std::cout << "PASS: grouped pass matches per-node pass\n";
    }
  }

  {
    // target_size > max_leaf: one walk per small subtree instead of per
    // leaf. Still exact at theta=0, and its larger target boxes make the MAC
    // more conservative, so at theta=0.5 it must not be less accurate than
    // the per-leaf walk of the per-node pass.
    auto particles = random_particles(2000, /*seed=*/6, /*spread=*/10.0);
    ForceParams fp;
    fp.eps2 = 1e-6;
    fp.G = 1.0;
    fp.use_quadrupole = false;
    fp.theta = 0.0;
    auto [p0, a0] = bh_test::run_bh(particles, fp, /*max_leaf=*/1, 64, /*target_size=*/32);
    double err0 = relative_error(a0, brute_force_accel(p0, fp.eps2, fp.G));
    fp.theta = 0.5;
    auto [pn, an] = bh_test::run_bh(particles, fp, /*max_leaf=*/1, 0);
    auto [pt, at] = bh_test::run_bh(particles, fp, /*max_leaf=*/1, 64, /*target_size=*/32);
    auto ref = brute_force_accel(pn, fp.eps2, fp.G);
    double err_node = relative_error(an, ref), err_target = relative_error(at, ref);
    std::cout << "target_size=32, max_leaf=1: theta=0 vs brute force " << err0 << ", theta=0.5 error " << err_target
              << " (per-leaf walk: " << err_node << ")\n";
    if (err0 > 1e-8 || err_target > err_node) {
      std::cerr << "FAIL: shared target walks are inexact or less accurate\n";
      rc = 1;
    } else {
      std::cout << "PASS: shared target walks\n";
    }
  }

  {
    // Large enough for flat_build_ttg.h's multi-chunk sample sort and its
    // backbone above the per-task subtrees (both only kick in above 4096
    // particles). theta=0 checks every particle is in exactly one leaf;
    // theta=0.5 against the per-node pass checks the backbone multipoles.
    // group_size 64 puts the groups inside the build subtrees, 8192 puts
    // them in the backbone.
    auto particles = random_particles(20000, /*seed=*/5, /*spread=*/10.0);
    ForceParams fp;
    fp.eps2 = 1e-6;
    fp.G = 1.0;
    fp.use_quadrupole = true;
    fp.theta = 0.5;
    auto [node_particles, node_accel] = bh_test::run_bh(particles, fp, /*max_leaf=*/16, 0);
    for (std::uint32_t gs : {64u, 8192u}) {
      fp.theta = 0.0;
      auto [p0, a0] = bh_test::run_bh(particles, fp, /*max_leaf=*/16, gs);
      double err0 = relative_error(a0, brute_force_accel(p0, fp.eps2, fp.G));
      fp.theta = 0.5;
      auto [p1, a1] = bh_test::run_bh(particles, fp, /*max_leaf=*/16, gs);
      double err1 = relative_error(a1, node_accel);
      std::cout << "20000 particles, group_size=" << gs << ": theta=0 vs brute force " << err0
                << ", theta=0.5 vs per-node " << err1 << "\n";
      if (err0 > 1e-8 || err1 > 1e-8) {
        std::cerr << "FAIL: parallel-built tree disagrees\n";
        rc = 1;
      } else {
        std::cout << "PASS: parallel-built tree matches\n";
      }
    }
  }

  ttg::fence();
  ttg::finalize();
  return rc;
}
