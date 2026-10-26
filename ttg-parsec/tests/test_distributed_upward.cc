// Verifies the distributed upward pass (src/distributed_upward.h): each
// simulated "rank" gets its own contiguous slice of particles (by Morton
// order) and its own Tree, wired together via ttg::Edge/aggregator. Run on
// one process (as ctest does), it simulates 5 ranks within it; under
// `mpirun -n P` with P > 1, it uses P ranks, one per process, so the
// exchange really goes between processes. Every process still builds all
// ranks' trees (only seeding its own), and the redistribution sort that
// would produce each rank's slice is not exercised. The reconstructed root
// (and skeleton) multipoles are compared against an ordinary single-tree
// upward pass over the SAME particles, which is the ground truth.
#include "../src/distributed_upward.h"
#include "../src/domain.h"
#include "../src/morton.h"
#include "../src/multipole.h"
#include "../src/particle.h"
#include "../src/tree.h"
#include "../src/types.h"
#include "../src/upward_ttg.h"

#include <ttg.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

using namespace bh;

namespace {

// Which (reference-depth) Morton cell `pos` falls into, given `domain`.
// Deliberately walks down level by level using the exact same
// center-comparison box_geometry()/octant_index() does (tree.cc), rather
// than a one-shot floor-division computation of the final-depth cell index:
// the two are mathematically equivalent in exact arithmetic, but floating-
// point rounding could disagree with the real tree build's own recursive
// classification for a particle sitting very close to a cell boundary -
// exactly the kind of mismatch that would silently assign a particle to
// the wrong rank. Walking the same way the tree itself does guarantees
// bit-for-bit agreement with wherever Tree::build_recursive actually puts
// this particle.
MortonCode particle_morton(const Vec3& pos, const Domain& domain) {
  Key key(0, {0, 0, 0});
  for (Level level = 0; level < kSortDepth; ++level) {
    Vec3 center = box_geometry(domain, key).first;
    unsigned oct = 0;
    if (pos.x >= center.x) oct |= 1u;
    if (pos.y >= center.y) oct |= 2u;
    if (pos.z >= center.z) oct |= 4u;
    key = key.child_at(static_cast<int>(oct));
  }
  return morton_lo(key);
}

double relative_error(const Multipole& a, const Multipole& b) {
  double num = std::abs(a.mass - b.mass) + norm(a.com - b.com);
  double den = std::abs(b.mass) + norm(b.com) + 1.0;
  for (int i = 0; i < 6; ++i) {
    num += std::abs(a.iab[static_cast<std::size_t>(i)] - b.iab[static_cast<std::size_t>(i)]);
    den += std::abs(b.iab[static_cast<std::size_t>(i)]);
  }
  return num / den;
}

}  // namespace

int main(int argc, char** argv) {
  ttg::initialize(argc, argv, -1);
  ttg::execute();
  int rc = 0;

  const std::size_t n = 4000;
  // simulated ranks: 5 within one process, or one per process (see
  // process_of in distributed_upward.h)
  const int nprocs = ttg::default_execution_context().size();
  const int nranks = nprocs > 1 ? nprocs : 5;
  const unsigned max_leaf = 8;

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> pos_dist(-10.0, 10.0);
  std::uniform_real_distribution<double> mass_dist(0.5, 1.5);
  std::vector<Particle> particles(n);
  for (std::size_t i = 0; i < n; ++i) {
    particles[i].id = static_cast<ParticleId>(i);
    particles[i].mass = mass_dist(rng);
    particles[i].pos = {pos_dist(rng), pos_dist(rng), pos_dist(rng)};
  }

  Domain domain = compute_domain(particles);

  // --- ground truth: ordinary, single-tree, single-rank upward pass ---
  Tree ground_tree(particles, max_leaf, domain);
  auto ground_up = make_upward_graph(ground_tree);
  ttg::make_graph_executable(ground_up.seed_leaf.get(), ground_up.combine_up.get());
  {  // on every process: the single-rank pass is replicated (see replicated.h)
    const auto& gparticles = ground_tree.particles();
    for (auto& [key, node] : ground_tree.nodes()) {
      if (!node.is_leaf()) continue;
      std::vector<Particle> slice(gparticles.begin() + node.particle_begin,
                                   gparticles.begin() + node.particle_begin + node.particle_count);
      ground_up.seed(key, std::move(slice));
    }
  }
  ttg::fence();

  Key root = ground_tree.root();
  if (ground_up.multipoles->data.find(root) == ground_up.multipoles->data.end()) {
    std::cerr << "FAIL: ground-truth upward pass never finalized the root\n";
    ttg::fence();
    ttg::finalize();
    return 1;
  }
  Multipole ground_root = ground_up.multipoles->data.at(root);

  // --- distributed: partition particles into `nranks` contiguous Morton
  // slices, build each rank's own local Tree, run the exchange ---
  std::vector<Particle> sorted = particles;
  std::sort(sorted.begin(), sorted.end(), [&](const Particle& a, const Particle& b) {
    return particle_morton(a.pos, domain) < particle_morton(b.pos, domain);
  });

  MortonCode full = MortonCode(1) << (NDIM * kSortDepth);
  std::vector<std::pair<MortonCode, MortonCode>> ranges(static_cast<std::size_t>(nranks));
  std::vector<std::vector<Particle>> per_rank(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) {
    std::size_t begin = n * static_cast<std::size_t>(r) / static_cast<std::size_t>(nranks);
    std::size_t end = n * static_cast<std::size_t>(r + 1) / static_cast<std::size_t>(nranks);
    auto& range = ranges[static_cast<std::size_t>(r)];
    range.first = (r == 0) ? 0 : particle_morton(sorted[begin].pos, domain);
    range.second = (r == nranks - 1) ? full : particle_morton(sorted[end].pos, domain);
    per_rank[static_cast<std::size_t>(r)].assign(sorted.begin() + static_cast<long>(begin),
                                                  sorted.begin() + static_cast<long>(end));
  }

  auto dist = make_distributed_upward(per_rank, domain, max_leaf, ranges);
  run_distributed_upward(dist);

  if (dist.multipoles->data.find(root) == dist.multipoles->data.end()) {
    std::cerr << "FAIL: distributed skeleton never reconstructed the global root\n";
    rc = 1;
  } else {
    Multipole dist_root = dist.multipoles->data.at(root);
    double err = relative_error(dist_root, ground_root);
    std::cout << "root multipole relative error: " << err << "\n";
    if (err > 1e-10) {
      std::cerr << "FAIL: distributed root multipole does not match the single-tree ground truth\n";
      rc = 1;
    }
  }

  // Not every key common to both maps is a meaningful comparison: a node
  // that's genuinely shared between ranks (its box extends into more than
  // one rank's slice) can still show up in dist.multipoles as an
  // INTERMEDIATE fold artifact - e.g. one rank's own subtree happened to
  // reach exactly that key before redirecting, while a neighboring rank had
  // so few particles nearby that its own tree never got that deep and
  // redirected from somewhere shallower instead. That key's value there is
  // real but incomplete; it gets superseded once folding reaches the true
  // shared boundary further up (which does end up with the complete,
  // correct total - see the root check, above). So only nodes genuinely,
  // fully owned by exactly one rank's declared range are guaranteed to hold
  // a final, complete value - compare just those (plus the root, which is
  // always final).
  int checked = 0, mismatched = 0;
  for (auto& [key, mp] : dist.multipoles->data) {
    bool owned_by_one_rank = key == root;
    for (int r = 0; !owned_by_one_rank && r < nranks; ++r) {
      if (range_contains(key, ranges[static_cast<std::size_t>(r)].first, ranges[static_cast<std::size_t>(r)].second)) {
        owned_by_one_rank = true;
      }
    }
    if (!owned_by_one_rank) continue;
    auto it = ground_up.multipoles->data.find(key);
    if (it == ground_up.multipoles->data.end()) continue;
    ++checked;
    if (relative_error(mp, it->second) > 1e-10) {
      ++mismatched;
      std::cerr << "FAIL: mismatch at key " << key << " dist.mass=" << mp.mass << " ground.mass=" << it->second.mass
                 << "\n";
    }
  }
  std::cout << "checked " << checked << " single-rank-owned nodes common to both builds, " << mismatched
             << " mismatched\n";
  if (mismatched > 0) {
    std::cerr << "FAIL: " << mismatched << " node(s) disagree between distributed and single-tree builds\n";
    rc = 1;
  }

  if (rc == 0) std::cout << "PASS: distributed upward pass matches single-tree ground truth\n";

  ttg::fence();
  ttg::finalize();
  return rc;
}
