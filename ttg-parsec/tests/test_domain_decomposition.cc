// Checks the domain-decomposed grouped pass (src/distributed_tree.h): run
// under `mpirun -n P`, every rank takes a slice of the same clustered
// particle set, repartitions it and computes forces from its locally
// essential tree. Every rank checks that its walks never opened a node
// they had no data for; rank 0 gathers the accelerations and checks that
// every particle was computed exactly once, and against a tree of all
// particles (built on every rank) and direct summation. The two trees
// differ where ranks' key ranges meet (shared cells are split further, and
// targets never span ranks), so forces differ by a small fraction of the
// Barnes-Hut error, and the error may grow a little. On one rank the
// decomposed tree is the single tree.
#include "../src/distributed_tree.h"
#include "../src/flat_build_ttg.h"
#include "../src/grouped_ttg.h"

#include <ttg.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <random>
#include <span>
#include <vector>

using namespace bh;

namespace {

struct IdAccel {
  std::uint32_t id;
  Vec3 a;
};

// Two dense clusters in a sparse background, so that ranks' key ranges
// cover far-apart regions (the case the LET's per-rank boxes are for).
std::vector<Particle> clustered_particles(std::size_t n) {
  std::mt19937 rng(11);
  std::normal_distribution<double> blob(0.0, 0.3);
  std::uniform_real_distribution<double> background(-10.0, 10.0), mass(0.5, 1.5);
  std::vector<Particle> particles(n);
  for (std::size_t i = 0; i < n; ++i) {
    Particle& p = particles[i];
    p.id = static_cast<ParticleId>(i);
    p.mass = mass(rng);
    if (i % 10 == 0) {
      p.pos = {background(rng), background(rng), background(rng)};
    } else {
      double c = (i % 2 == 0) ? -4.0 : 4.0;
      p.pos = {c + blob(rng), c + blob(rng), blob(rng)};
    }
  }
  return particles;
}

}  // namespace

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();
  const int nranks = num_ranks(), me = this_rank();
  const std::size_t n = 30000;
  const unsigned max_leaf = 8;
  const std::uint32_t group_size = 256, target_size = 32;
  ForceParams fp;
  fp.theta = 0.5;
  fp.eps2 = 1e-6;
  fp.G = 1.0;
  fp.use_quadrupole = false;

  const std::vector<Particle> all = clustered_particles(n);

  // decomposed: this rank's slice, repartitioned
  std::vector<Particle> local(all.begin() + n * me / nranks, all.begin() + n * (me + 1) / nranks);
  FlatTree f;
  BuildScratch ws;
  build_distributed_tree(f, ws, local, max_leaf, group_size, fp.theta);
  std::vector<IdAccel> mine;
  std::mutex mutex;
  std::uint64_t missing = run_grouped_pass(f, fp, target_size, [&](Particle& p, const Vec3& a) {
    std::lock_guard<std::mutex> lock(mutex);
    mine.push_back({p.id, a});
  });
  int rc = 0;
  if (missing != 0) {
    std::cerr << "FAIL: rank " << me << "'s walks opened " << missing << " nodes its locally essential tree lacks\n";
    rc = 1;
  }
  std::vector<std::span<const IdAccel>> to_root(nranks);
  to_root[0] = mine;
  auto gathered = exchange<IdAccel>(to_root);

  // a tree of all particles, on every rank (run_grouped_pass is collective)
  std::vector<Vec3> single(n);
  FlatTree whole = build_flat_tree(all, max_leaf, group_size);
  run_grouped_pass(whole, fp, target_size, [&](Particle& p, const Vec3& a) { single[p.id] = a; });

  if (me == 0) {
    std::vector<Vec3> decomposed(n);
    std::vector<int> times(n, 0);
    for (const auto& part : gathered) {
      for (const IdAccel& r : part) {
        decomposed[r.id] = r.a;
        ++times[r.id];
      }
    }
    std::size_t wrong_count = 0;
    for (int t : times) wrong_count += (t != 1);
    double diff = 0.0, norm_single = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      diff += norm2(decomposed[i] - single[i]);
      norm_single += norm2(single[i]);
    }
    double rel_diff = std::sqrt(diff / norm_single);
    // error against direct summation, for every 15th particle
    double err_decomposed = 0.0, err_single = 0.0, norm_direct = 0.0;
    for (std::size_t i = 0; i < n; i += 15) {
      Vec3 a{};
      for (const Particle& q : all) {
        if (q.id == all[i].id) continue;
        Vec3 d = q.pos - all[i].pos;
        double r2 = norm2(d) + fp.eps2;
        a += (q.mass / (r2 * std::sqrt(r2))) * d;
      }
      err_decomposed += norm2(decomposed[i] - a);
      err_single += norm2(single[i] - a);
      norm_direct += norm2(a);
    }
    err_decomposed = std::sqrt(err_decomposed / norm_direct);
    err_single = std::sqrt(err_single / norm_direct);
    std::cout << nranks << " ranks: " << wrong_count << " particles not computed exactly once; decomposed vs single tree "
              << rel_diff << "; error vs direct: decomposed " << err_decomposed << ", single tree " << err_single << "\n";
    if (wrong_count != 0) {
      std::cerr << "FAIL: some particles were not computed exactly once\n";
      rc = 1;
    }
    if (rel_diff > 0.2 * err_single) {
      std::cerr << "FAIL: decomposed forces differ from the single tree by more than the shared cells explain\n";
      rc = 1;
    }
    if (err_decomposed > 1.05 * err_single + 1e-12) {
      std::cerr << "FAIL: decomposed forces are less accurate than the single tree's\n";
      rc = 1;
    }
    if (rc == 0) std::cout << "PASS: decomposed forces match\n";
  }

  ttg::fence();
  ttg::finalize();
  return rc;
}
