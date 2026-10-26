#pragma once

#include "domain.h"
#include "morton.h"
#include "multipole.h"
#include "particle.h"
#include "replicated.h"
#include "tree.h"
#include "types.h"

#include <ttg.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bh {

// A RawMoment a rank could not finish combining locally - the parent it was
// headed for lies outside that rank's own slice of the Morton order (see
// RankUpward::combine_up, below) - bundled up and exchanged between ranks
// (see ExchangeGraph) rather than routed through whichever rank happens to
// "own" that shared ancestor. Serializes explicitly (see Particle/Vec3/Key
// for the same pattern) since it flows through a TTG edge.
struct Contribution {
  Key key;
  RawMoment moment;

  template <typename Archive>
  void serialize(Archive& ar) { ar & key & moment; }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & key & moment; }
};

// One rank's own local upward pass over the particles it owns - a
// contiguous slice [lo, hi) of the global Morton order (see morton.h).
// Structurally almost identical to make_upward_graph (upward_ttg.h):
// seed_leaf seeds from this rank's own particles, combine_up aggregates
// bottom-up. The one difference is combine_up's redirect: when a node's
// PARENT is still entirely inside [lo, hi) (range_contains), it keeps
// combining locally exactly as upward_ttg.h does; otherwise - this rank has
// reached the shallowest node it can finish alone - it publishes its
// contribution via branch_edge (keyed by this rank's own index) instead of
// self-looping, for the shared ExchangeGraph (below) to collect.
//
// Kept as its own small, self-contained implementation rather than adding
// an optional "distributed" mode to make_upward_graph, so the existing,
// tested single-rank path in upward_ttg.h is never put at risk by this.
//
// The "ranks" here are simulated ranks; simulated rank r runs on process
// r % (number of processes) (process_of), so with one process all of them
// share it (tests/test_distributed_upward.cc's default) and with as many
// processes as simulated ranks each runs on its own, exchanging the
// branch contributions between processes.
struct RankUpward {
  ttg::Edge<Key, std::vector<Particle>> seed_edge;
  ttg::Edge<Key, RawMoment> raw_edge;
  std::unique_ptr<ttg::TTBase> seed_leaf;
  std::unique_ptr<ttg::TTBase> combine_up;
  std::function<void(Key, std::vector<Particle>)> seed;
  std::shared_ptr<LeafParticleStore> leaf_particles = std::make_shared<LeafParticleStore>();
};

// The process simulated rank `rank` runs on.
inline int process_of(int rank) { return rank % num_ranks(); }

inline RankUpward make_rank_upward(Tree& tree, int rank, MortonCode lo, MortonCode hi,
                                    std::shared_ptr<MultipoleStore> multipoles,
                                    ttg::Edge<int, Contribution>& branch_edge) {
  RankUpward g;
  Key root = tree.root();

  auto seed_leaf_tt = ttg::make_tt(
      [&tree, leaf_particles = g.leaf_particles, rank, lo, hi](const Key& leaf_key,
                                                                const std::vector<Particle>& particles) {
        {
          std::lock_guard<std::mutex> lock(leaf_particles->mutex);
          leaf_particles->data[leaf_key] = particles;
        }
        if (leaf_key == tree.root()) return;  // this rank's whole range is a single leaf
        Key parent_key = leaf_key.parent();
        const Vec3& ref = tree.box_geometry(parent_key).first;
        RawMoment rm;
        for (const auto& p : particles) rm += raw_moment_of_particle(p.mass, p.pos, ref);
        // A leaf can itself be the exact boundary (its own box fully local,
        // but its parent's isn't) just as well as an interior node can -
        // see combine_up_tt's identical check, below.
        if (range_contains(parent_key, lo, hi)) {
          ttg::send<0>(parent_key, rm);
        } else {
          ttg::send<1>(rank, Contribution{parent_key, rm});
        }
      },
      ttg::edges(g.seed_edge), ttg::edges(g.raw_edge, branch_edge), "rank_seed_leaf");

  auto target_fn = [&tree](const Key& n) -> std::size_t { return tree.num_children(n); };
  auto agg_edge = ttg::make_aggregator(g.raw_edge, target_fn);

  auto combine_up_tt = ttg::make_tt(
      [&tree, root, multipoles, rank, lo, hi](const Key& key, const ttg::Aggregator<RawMoment>& agg) {
        RawMoment sum;
        for (auto&& v : agg) sum += v;
        Multipole multipole = finalize_raw_moment(sum, tree.box_geometry(key).first);
        {
          std::lock_guard<std::mutex> lock(multipoles->mutex);
          multipoles->data[key] = multipole;
        }
        if (key == root) return;  // this rank's local tree root - nothing left to exchange

        Key parent_key = key.parent();
        RawMoment shifted = raw_moment_from_multipole(multipole, tree.box_geometry(parent_key).first);
        if (range_contains(parent_key, lo, hi)) {
          ttg::send<0>(parent_key, shifted);                       // still local: keep combining
        } else {
          ttg::send<1>(rank, Contribution{parent_key, shifted});   // shared: publish for the exchange
        }
      },
      ttg::edges(agg_edge), ttg::edges(g.raw_edge, branch_edge), "rank_combine_up");

  auto* seed_terminal = seed_leaf_tt->template in<0>();
  g.seed = [seed_terminal](Key key, std::vector<Particle> particles) {
    seed_terminal->send(key, std::move(particles));
  };

  seed_leaf_tt->set_keymap([rank](const Key&) { return process_of(rank); });
  combine_up_tt->set_keymap([rank](const Key&) { return process_of(rank); });

  g.seed_leaf = std::move(seed_leaf_tt);
  g.combine_up = std::move(combine_up_tt);
  return g;
}

// Collects every rank's branch-edge contributions - bundled into exactly
// ONE message per rank, since each rank's expected contribution count is
// known in advance by walking its own built tree (see make_distributed_upward)
// - and broadcasts the complete set to every receiver, so the allgather step
// always waits for exactly `nranks` messages, never a per-node count that
// would vary with how many ranks happen to share any given branch node.
// Every receiver then finishes the same bottom-up combination
// independently and locally: once every rank's contribution is visible
// everywhere, no further communication is needed to reconstruct the
// identical root-to-branch skeleton on every rank.
struct ExchangeGraph {
  ttg::Edge<int, Contribution> branch_edge;                 // keyed by SENDING rank
  ttg::Edge<int, std::vector<Contribution>> exchange_edge;  // keyed by RECEIVING rank
  ttg::Edge<Key, void> done_edge;                           // fires once per receiver, with the root
  std::unique_ptr<ttg::TTBase> bundle;
  std::unique_ptr<ttg::TTBase> allgather;
};

inline ExchangeGraph make_exchange_graph(const Domain& domain, Key root, int nranks,
                                          std::vector<std::size_t> expected_per_rank,
                                          std::shared_ptr<MultipoleStore> multipoles) {
  ExchangeGraph g;

  auto bundle_target_fn = [expected_per_rank](const int& rank) -> std::size_t {
    return expected_per_rank[static_cast<std::size_t>(rank)];
  };
  auto bundle_agg = ttg::make_aggregator(g.branch_edge, bundle_target_fn);

  auto bundle_tt = ttg::make_tt(
      [nranks](const int&, const ttg::Aggregator<Contribution>& agg) {
        std::vector<Contribution> bundle(agg.begin(), agg.end());
        for (int r = 0; r < nranks; ++r) ttg::send<0>(r, bundle);
      },
      ttg::edges(bundle_agg), ttg::edges(g.exchange_edge), "exchange_bundle");

  auto allgather_target_fn = [nranks](const int&) -> std::size_t { return static_cast<std::size_t>(nranks); };
  auto allgather_agg = ttg::make_aggregator(g.exchange_edge, allgather_target_fn);

  auto allgather_tt = ttg::make_tt(
      [domain, root, multipoles](const int&, const ttg::Aggregator<std::vector<Contribution>>& agg) {
        std::unordered_map<Key, RawMoment> combined;
        for (auto&& bundle : agg) {
          for (const auto& c : bundle) combined[c.key] += c.moment;
        }

        // Finite, purely local bottom-up fold over the now-complete set of
        // branch contributions - unlike the live per-rank combine_up, this
        // needs no aggregator/target count: every contribution that will
        // ever exist for this round is already in `combined`.
        //
        // Different ranks' redirect boundaries can sit at DIFFERENT depths
        // along the same shared ancestor chain (one rank's local territory
        // might run deeper than another's before hitting shared ground), so
        // this can't just fold level-by-level in lockstep - a shallower
        // direct contribution could get finalized before a deeper one that
        // was still due to fold up into it. Instead, always finalize the
        // DEEPEST key still present first: nothing else in `combined` can
        // ever be its descendant (different ranks' redirect points can't
        // nest inside one another, since each is built from a disjoint
        // slice of particles), so whatever's deepest right now is always
        // safe to finalize, and its contribution is merged into its
        // parent's entry (new or already-pending) before that parent can
        // ever be picked.
        while (!combined.empty()) {
          auto it = std::max_element(combined.begin(), combined.end(), [](const auto& a, const auto& b) {
            return a.first.level() < b.first.level();
          });
          Key key = it->first;
          RawMoment rm = it->second;
          combined.erase(it);

          Multipole mp = finalize_raw_moment(rm, box_geometry(domain, key).first);
          {
            std::lock_guard<std::mutex> lock(multipoles->mutex);
            multipoles->data[key] = mp;
          }
          if (key == root) {
            ttg::sendk<0>(root);
            return;
          }
          Key parent = key.parent();
          combined[parent] += raw_moment_from_multipole(mp, box_geometry(domain, parent).first);
        }
      },
      ttg::edges(allgather_agg), ttg::edges(g.done_edge), "exchange_allgather");

  // bundles are keyed by the sending simulated rank, allgathers by the
  // receiving one
  bundle_tt->set_keymap([](const int& rank) { return process_of(rank); });
  allgather_tt->set_keymap([](const int& rank) { return process_of(rank); });

  g.bundle = std::move(bundle_tt);
  g.allgather = std::move(allgather_tt);
  return g;
}

// Ties together N ranks' local upward passes and the exchange that
// reconstructs the shared skeleton on every one of them. `particles_per_rank`
// and `ranges` must already reflect each rank's contiguous slice of the
// global Morton order - the actual parallel redistribution sort that would
// produce that partitioning in a real deployment is a separate problem, not
// handled here.
struct DistributedUpward {
  std::vector<std::unique_ptr<Tree>> trees;  // unique_ptr: Tree holds internal state make_rank_upward
                                              // captures by reference, so these must never move once built
  std::vector<RankUpward> ranks;
  ExchangeGraph exchange;
  std::shared_ptr<MultipoleStore> multipoles = std::make_shared<MultipoleStore>();
};

inline DistributedUpward make_distributed_upward(
    std::vector<std::vector<Particle>> particles_per_rank, const Domain& domain,
    unsigned max_particles_per_leaf, const std::vector<std::pair<MortonCode, MortonCode>>& ranges) {
  int nranks = static_cast<int>(particles_per_rank.size());
  DistributedUpward d;
  d.trees.reserve(static_cast<std::size_t>(nranks));
  for (auto& p : particles_per_rank) {
    d.trees.push_back(std::make_unique<Tree>(std::move(p), max_particles_per_leaf, domain));
  }

  // How many "my parent is shared" redirects each rank's upward pass will
  // actually produce. NOT decompose_range(lo, hi).size(): that counts the
  // geometrically-maximal blocks the RANGE decomposes into, assuming every
  // one of them holds at least one particle - but most of a sparse,
  // non-uniform particle distribution's key-space is empty, and a node with
  // no particles under it is never built at all, so it can never redirect.
  //
  // Nor is it simply "every real node whose parent is non-local": once an
  // INTERIOR node is itself non-local, all of its children see that same
  // non-local parent and redirect past it uniformly, so that node's own
  // aggregator never receives anything and never fires - it produces no
  // redirect of its own, even though it undeniably exists and its parent is
  // indeed shared. An interior node only actually fires (and can then
  // redirect) if it is ITSELF local - that locality is exactly what let all
  // of its children agree to send to it in the first place.
  //
  // A LEAF is different: it fires unconditionally (seed_leaf isn't gated by
  // an aggregator the way combine_up is), and a leaf's own box is sized by
  // particle count, not alignment with any rank's range - so a leaf can
  // itself be non-local (its particles are few enough to stop recursion
  // early, even though its box still spans into a neighboring rank's
  // territory) and still always redirect via its parent's (non-)locality.
  std::vector<std::size_t> expected(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) {
    const Tree& tree = *d.trees[static_cast<std::size_t>(r)];
    auto [lo, hi] = ranges[static_cast<std::size_t>(r)];
    std::size_t count = 0;
    for (auto& [key, node] : tree.nodes()) {
      if (key == tree.root()) continue;  // no parent to redirect to
      if (range_contains(key.parent(), lo, hi)) continue;  // parent is local: sends normally, no redirect
      if (node.is_leaf() || range_contains(key, lo, hi)) ++count;
    }
    expected[static_cast<std::size_t>(r)] = count;
  }

  d.exchange = make_exchange_graph(domain, d.trees.front()->root(), nranks, std::move(expected), d.multipoles);

  for (int r = 0; r < nranks; ++r) {
    auto [lo, hi] = ranges[static_cast<std::size_t>(r)];
    d.ranks.push_back(
        make_rank_upward(*d.trees[static_cast<std::size_t>(r)], r, lo, hi, d.multipoles, d.exchange.branch_edge));
  }
  return d;
}

// Makes every rank's graph (and the shared exchange) executable, seeds each
// rank from its own tree's leaves, and fences. Must be called after
// ttg::execute().
//
// Uses TTBase::make_executable() directly in a loop rather than
// ttg::make_graph_executable() (upward_ttg.h/bh_pass.h), since that helper
// is a fixed-arity variadic and the rank count here is only known at
// runtime. That skips the backend's startup hook the helper calls, which
// is fine across processes too: the PaRSEC backend holds on to messages for
// a TT this process has not made executable yet until it is.
inline void run_distributed_upward(DistributedUpward& d) {
  for (auto& r : d.ranks) {
    r.seed_leaf->make_executable();
    r.combine_up->make_executable();
  }
  d.exchange.bundle->make_executable();
  d.exchange.allgather->make_executable();

  for (std::size_t r = 0; r < d.ranks.size(); ++r) {
    if (process_of(static_cast<int>(r)) != this_rank()) continue;
    Tree& tree = *d.trees[r];
    const auto& particles = tree.particles();
    for (auto& [key, node] : tree.nodes()) {
      if (!node.is_leaf()) continue;
      std::vector<Particle> slice(particles.begin() + node.particle_begin,
                                   particles.begin() + node.particle_begin + node.particle_count);
      d.ranks[r].seed(key, std::move(slice));
    }
  }
  ttg::fence();
}

}  // namespace bh
