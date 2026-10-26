#pragma once

#include "domain.h"
#include "flat_tree.h"
#include "particle.h"
#include "ttg_parallel_for.h"
#include "types.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace bh {

// Parallel construction of a FlatTree, as a sequence of ttg_parallel_for
// phases (one fence each):
//   1. bounding box per chunk of particles, combined into the Domain;
//   2. Morton key per particle, each chunk sorting its own (key, index) pairs;
//   3. sample sort: splitters from regular samples of every sorted chunk,
//      each chunk locating its bucket boundaries;
//   4. one task per bucket gathers its pieces from every chunk, sorts them
//      and writes the particles in Morton order (AoS and SoA);
//   5. a serial backbone splits the top of the tree breadth-first down to
//      subtrees of at most kSubtreeCutoff particles;
//   6. one task per subtree counts its nodes (the same splits, nothing
//      written), which lays the subtrees out behind the backbone;
//   7. one task per subtree builds it, and its multipoles, in place;
//   8. the backbone's multipoles are finalized serially, bottom-up.
// Building in place, rather than into per-subtree buffers that are then
// copied, halves the build's memory traffic at max_leaf 1, where the tree
// has ~1.5 nodes per particle.
// The multipoles are complete when this returns, so the grouped pass needs
// no separate upward pass. Phase 1 (particle_bounds), phases 2-4
// (sort_particles) and phases 5-8 (build_nodes) are also callable on their
// own, for the domain decomposition (distributed_tree.h), which agrees on
// the domain and repartitions the particles between them.
//
// The FlatTree and the BuildScratch are reused from step to step: at 32M
// particles, allocating, first-touching and freeing their ~10 GB anew
// every step cost more than 2 s per step on 128 threads (page faults and
// munmap are largely serial), against ~5 s for the walk itself. Every
// array is cleared and resized, never shrunk, so after the first step
// nothing is allocated unless the tree grows.

namespace detail {

// Subtrees with at most this many particles are built by one task (the
// same cutoff the HPX code of the AMT-BH benchmark uses). Independent of
// group_size, which only sets the granularity of the force walk.
constexpr std::uint32_t kSubtreeCutoff = 4096;

// Chunks for phases 1-4: enough to balance, few enough that the P^2
// bucket-boundary table stays small.
inline std::uint32_t num_sort_chunks(std::size_t n) {
  return static_cast<std::uint32_t>(std::clamp<std::size_t>(n / 4096, 1, 1024));
}

// A particle's Morton key and its index in the input; ordered by key, ties
// by index, so the sort is deterministic.
struct KeyIndex {
  std::uint64_t key;
  std::uint32_t index;
  bool operator<(const KeyIndex& o) const { return key < o.key || (key == o.key && index < o.index); }
};

}  // namespace detail

// Per-step working storage of build_flat_tree, kept by the caller so its
// capacity carries over to the next step.
struct BuildScratch {
  UninitVector<detail::KeyIndex> chunked, sorted;
  UninitVector<std::uint64_t> keys;
  std::vector<NodeIndex> subtree_root;
  std::vector<std::uint32_t> subtree_count;
  std::vector<NodeIndex> subtree_base;
  std::vector<std::vector<NodeIndex>> subtree_groups;
  // domain decomposition only (distributed_tree.h)
  UninitVector<std::uint64_t> route_keys;
  UninitVector<Particle> routed;
};

// clear() + resize() rather than resize(): a growing vector then reallocates
// without copying stale contents, and an UninitVector stays uninitialized.
template <typename V>
void reset(V& v, std::size_t n) {
  v.clear();
  v.resize(n);
}

// Phase 1: the bounding box (lower, upper corner) of `input`.
inline std::pair<Vec3, Vec3> particle_bounds(const std::vector<Particle>& input) {
  const auto n = static_cast<std::uint32_t>(input.size());
  const std::uint32_t P = detail::num_sort_chunks(n);
  auto chunk_begin = [n, P](std::uint32_t c) { return static_cast<std::uint32_t>(std::uint64_t(n) * c / P); };
  constexpr double inf = std::numeric_limits<double>::infinity();
  std::vector<std::pair<Vec3, Vec3>> boxes(P, {Vec3{inf, inf, inf}, Vec3{-inf, -inf, -inf}});
  ttg_parallel_for(P, [&](std::uint32_t c) {
    auto& [lo, hi] = boxes[c];
    for (std::uint32_t i = chunk_begin(c); i < chunk_begin(c + 1); ++i) {
      const Vec3& p = input[i].pos;
      lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
      hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
  });
  Vec3 lo = boxes[0].first, hi = boxes[0].second;
  for (const auto& [l, h] : boxes) {
    lo = {std::min(lo.x, l.x), std::min(lo.y, l.y), std::min(lo.z, l.z)};
    hi = {std::max(hi.x, h.x), std::max(hi.y, h.y), std::max(hi.z, h.z)};
  }
  return {lo, hi};
}

// Phases 2-4: sorts `input` by Morton key in `domain` into f.particles
// (and f.x/y/z/m), with the sorted keys in ws.keys.
inline void sort_particles(FlatTree& f, BuildScratch& ws, const std::vector<Particle>& input, const Domain& domain) {
  const auto n = static_cast<std::uint32_t>(input.size());
  const std::uint32_t P = detail::num_sort_chunks(n);
  auto chunk_begin = [n, P](std::uint32_t c) { return static_cast<std::uint32_t>(std::uint64_t(n) * c / P); };
  f.domain = domain;

  // 2. keys, sorted per chunk
  using detail::KeyIndex;
  auto& chunked = ws.chunked;
  reset(chunked, n);
  ttg_parallel_for(P, [&](std::uint32_t c) {
    for (std::uint32_t i = chunk_begin(c); i < chunk_begin(c + 1); ++i) chunked[i] = {morton_key(f.domain, input[i].pos), i};
    std::sort(chunked.begin() + chunk_begin(c), chunked.begin() + chunk_begin(c + 1));
  });

  // 3. splitters from kSamples regular samples per chunk; bucket b holds
  //    keys in [splitter[b-1], splitter[b])
  constexpr std::uint32_t kSamples = 16;
  std::vector<KeyIndex> samples;
  samples.reserve(std::size_t(P) * kSamples);
  for (std::uint32_t c = 0; c < P; ++c) {
    std::uint32_t b = chunk_begin(c), len = chunk_begin(c + 1) - b;
    for (std::uint32_t s = 0; s < kSamples && len > 0; ++s) samples.push_back(chunked[b + std::uint64_t(len) * s / kSamples]);
  }
  std::sort(samples.begin(), samples.end());
  std::vector<KeyIndex> splitters(P - 1);
  for (std::uint32_t b = 1; b < P; ++b) splitters[b - 1] = samples[std::size_t(samples.size()) * b / P];
  // bounds[c * (P + 1) + b] = start of bucket b within chunk c
  std::vector<std::uint32_t> bounds(std::size_t(P) * (P + 1));
  ttg_parallel_for(P, [&](std::uint32_t c) {
    std::uint32_t* bc = &bounds[std::size_t(c) * (P + 1)];
    auto first = chunked.begin() + chunk_begin(c), last = chunked.begin() + chunk_begin(c + 1);
    bc[0] = chunk_begin(c);
    for (std::uint32_t b = 1; b < P; ++b) {
      bc[b] = static_cast<std::uint32_t>(std::lower_bound(first, last, splitters[b - 1]) - chunked.begin());
    }
    bc[P] = chunk_begin(c + 1);
  });
  std::vector<std::uint32_t> bucket_begin(P + 1, 0);
  for (std::uint32_t b = 0; b < P; ++b) {
    std::uint32_t size = 0;
    for (std::uint32_t c = 0; c < P; ++c) size += bounds[std::size_t(c) * (P + 1) + b + 1] - bounds[std::size_t(c) * (P + 1) + b];
    bucket_begin[b + 1] = bucket_begin[b] + size;
  }

  // 4. gather + sort each bucket, write particles in Morton order
  auto& sorted = ws.sorted;
  auto& keys = ws.keys;
  reset(sorted, n);
  reset(keys, n);
  if (f.particles.size() != n) f.particles.resize(n);  // fully overwritten below
  reset(f.x, n);
  reset(f.y, n);
  reset(f.z, n);
  reset(f.m, n);
  ttg_parallel_for(P, [&](std::uint32_t b) {
    std::uint32_t out = bucket_begin[b];
    for (std::uint32_t c = 0; c < P; ++c) {
      const std::uint32_t* bc = &bounds[std::size_t(c) * (P + 1)];
      out = static_cast<std::uint32_t>(std::copy(chunked.begin() + bc[b], chunked.begin() + bc[b + 1], sorted.begin() + out) -
                                       sorted.begin());
    }
    std::sort(sorted.begin() + bucket_begin[b], sorted.begin() + bucket_begin[b + 1]);
    for (std::uint32_t i = bucket_begin[b]; i < bucket_begin[b + 1]; ++i) {
      const Particle& p = input[sorted[i].index];
      keys[i] = sorted[i].key;
      f.particles[i] = p;
      f.x[i] = p.pos.x;
      f.y[i] = p.pos.y;
      f.z[i] = p.pos.z;
      f.m[i] = p.mass;
    }
  });
}

// Phases 5-8: the nodes over the particles sort_particles left in f and
// ws.keys, rooted at the domain's cell. No nodes at all without particles.
inline void build_nodes(FlatTree& f, BuildScratch& ws, const LeafRule& rule, std::uint32_t group_size) {
  f.groups.clear();
  f.root = 0;
  const auto n = static_cast<std::uint32_t>(f.particles.size());
  auto& keys = ws.keys;

  // 5. backbone: breadth-first from the root, directly into f, down to the
  //    subtree roots
  auto& subtree_root = ws.subtree_root;
  subtree_root.clear();
  auto is_subtree_root = [&](NodeIndex i) {
    return f.nodes[i].particle_count <= detail::kSubtreeCutoff || f.nodes[i].leaf;
  };
  auto grow = [&](std::size_t size) {
    f.nodes.resize(size);
    f.center.resize(size);
    f.iab.resize(size);
  };
  f.nodes.clear();  // clear() keeps the capacity of the previous step
  f.center.clear();
  f.iab.clear();
  if (n == 0) return;
  grow(1);
  detail::init_node(f, 0, 0, n, f.domain.center, f.domain.half_width, 0, keys.data(), rule);
  if (detail::is_group_root(n, f.nodes[0].leaf, std::numeric_limits<std::uint32_t>::max(), group_size)) {
    f.groups.push_back(0);
  }
  for (NodeIndex i = 0; i < f.nodes.size(); ++i) {
    if (is_subtree_root(i)) {
      subtree_root.push_back(i);
      continue;
    }
    auto first = static_cast<NodeIndex>(f.nodes.size());
    grow(first + 8);  // room for up to 8 children
    grow(first + detail::split_node(f, i, first, keys.data(), rule, group_size, f.groups));
  }
  const auto backbone_size = static_cast<NodeIndex>(f.nodes.size());
  const auto num_subtrees = static_cast<std::uint32_t>(subtree_root.size());

  // 6. each subtree's node count, to lay the subtrees out behind the backbone
  auto& count = ws.subtree_count;
  reset(count, num_subtrees);
  ttg_parallel_for(num_subtrees, [&](std::uint32_t s) {
    std::vector<detail::KeyRange> stack;
    const WalkNode& r = f.nodes[subtree_root[s]];
    count[s] = detail::count_subtree(keys.data(), r.particle_begin, r.particle_begin + r.particle_count, r.depth, rule,
                                     stack);
  });
  auto& base = ws.subtree_base;  // subtree s's non-root nodes start here
  reset(base, num_subtrees + 1);
  base[0] = backbone_size;
  for (std::uint32_t s = 0; s < num_subtrees; ++s) base[s + 1] = base[s] + count[s] - 1;
  const NodeIndex total = base[num_subtrees];
  if (f.nodes.capacity() < total) {
    // grow with slack, so a slightly larger tree next step does not
    // reallocate (and re-fault) the node arrays again
    std::size_t cap = total + total / 8;
    f.nodes.reserve(cap);
    f.center.reserve(cap);
    f.iab.reserve(cap);
  }
  grow(total);

  // 7. subtrees, in parallel, in place
  auto& subtree_groups = ws.subtree_groups;
  if (subtree_groups.size() < num_subtrees) subtree_groups.resize(num_subtrees);  // never shrink: keeps the inner capacity
  ttg_parallel_for(num_subtrees, [&](std::uint32_t s) {
    subtree_groups[s].clear();
    detail::build_subtree(f, subtree_root[s], base[s], keys.data(), rule, group_size, subtree_groups[s]);
  });

  // 8. backbone multipoles, bottom-up (children follow their parent, so
  //    reverse order sees every child first; subtree roots are done)
  for (NodeIndex i = backbone_size; i-- > 0;) {
    if (!is_subtree_root(i)) detail::finalize_node(f, i);
  }
  for (std::uint32_t s = 0; s < num_subtrees; ++s) {
    f.groups.insert(f.groups.end(), subtree_groups[s].begin(), subtree_groups[s].end());
  }
}

// Builds the tree of `input` into `f`, reusing the storage of `f` and `ws`.
inline void build_flat_tree(FlatTree& f, BuildScratch& ws, const std::vector<Particle>& input, unsigned max_leaf,
                            std::uint32_t group_size) {
  auto [lo, hi] = particle_bounds(input);
  sort_particles(f, ws, input, domain_from_bounds(lo, hi));
  build_nodes(f, ws, LeafRule{max_leaf}, group_size);
}

// One-off build with temporary storage (tests, tools).
inline FlatTree build_flat_tree(const std::vector<Particle>& input, unsigned max_leaf, std::uint32_t group_size) {
  FlatTree f;
  BuildScratch ws;
  build_flat_tree(f, ws, input, max_leaf, group_size);
  return f;
}

}  // namespace bh
