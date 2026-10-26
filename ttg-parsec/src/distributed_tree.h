#pragma once

#include "domain.h"
#include "exchange_ttg.h"
#include "flat_build_ttg.h"
#include "flat_tree.h"
#include "multipole.h"
#include "particle.h"
#include "types.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

// Domain decomposition of the grouped pass across ranks, with a locally
// essential tree (LET, Salmon 1991; Warren & Salmon 1993), as PEPC does:
//
//  1. Every rank's particle bounding box is gathered into the global
//     Domain, so all ranks agree on the cells (and Morton keys).
//  2. The ranks sort their particles by key and pick splitters from
//     weighted samples of every rank's keys, so each rank owns one
//     contiguous key range holding ~N/P particles; particles move to their
//     owner (an all-to-all) and are sorted again.
//  3. Each rank builds the tree of its own particles (build_nodes), but
//     never stops splitting a cell that reaches into another rank's range
//     (LeafRule), so its particles all lie below its *branches*: the
//     topmost cells entirely inside its range, whose subtrees are exactly
//     what a tree of all particles would have there.
//  4. Every rank gathers all ranks' branches (cell, multipole, and the
//     bounding box of the branch's particles), and assembles the same *top
//     tree* from them, behind its local nodes: copies of its own branches
//     lead into its local subtrees, the other ranks' branches are
//     placeholders.
//  5. For every other rank q, each rank walks its branches against the
//     boxes of q's branches with the walk's own MAC and sends q what q's
//     walks can open: every node not accepted for all of q's boxes, with
//     its children, and the particles of such leaves. q grafts these below
//     its placeholders. A node accepted for all of q's boxes is accepted by
//     every target of q too - each target's box lies inside one of them -
//     so q's walks never open a node they do not have. One box per branch
//     rather than one per rank, because a key range can cover far-apart
//     regions (both ends of a Morton curve segment), whose common box
//     would accept little; a hierarchy over the boxes (BoxHierarchy) keeps
//     the test cheap.
//
// The grouped pass then walks the local groups as on a single rank, from
// the top tree's root, and kicks/drifts the local particles, which become
// the next step's input. The tree differs from a single-rank tree only in
// cells shared between ranks (split further here, where the single tree
// might have stopped at max_leaf), so forces agree to within the
// Barnes-Hut error, not to roundoff.

namespace bh {

// A rank's particle count and bounding box, gathered by every rank.
struct RankBox {
  Vec3 lo, hi;
  std::uint64_t count = 0;
};

// A branch: a topmost cell entirely inside its owner's key range.
struct BranchRecord {
  std::uint64_t key = 0;  // the cell's first key
  Vec3 lo, hi;            // bounding box of its particles
  Vec3 com;
  double mass = 0.0;
  Sym3 iab{};
  std::uint32_t count = 0;
  NodeIndex local = 0;  // its node on the owning rank
  std::int32_t owner = 0;
  std::uint8_t depth = 0;
  std::uint8_t leaf = 0;
};

// A node of a pruned subtree sent to another rank (see step 5 above);
// first_child and particle_begin are relative to the sent arrays. Records
// with branch != kNoBranch stand for that (global) branch itself.
struct LetNode {
  static constexpr std::uint32_t kNoBranch = std::numeric_limits<std::uint32_t>::max();
  Vec3 com;
  double mass = 0.0;
  double half_width = 0.0;
  Sym3 iab{};
  std::uint32_t first_child = 0, particle_begin = 0, particle_count = 0;
  std::uint32_t branch = kNoBranch;
  std::uint8_t num_children = 0;
  std::uint8_t leaf = 0;
};

struct LetParticle {
  double x, y, z, m;
};

// Wall time of build_distributed_tree: in collective exchanges, and the rest.
struct DistributedBuildTiming {
  double build_ms = 0.0, exchange_ms = 0.0;
};

namespace detail {

// A binary hierarchy over a rank's branch boxes (in key order, so nearby
// boxes share subtrees), for the LET's test: whether a source node is
// accepted for every box - true as soon as a subtree's common box accepts
// it, false as soon as a single box does not.
class BoxHierarchy {
 public:
  explicit BoxHierarchy(const std::vector<BBox>& boxes) {
    if (!boxes.empty()) build(boxes, 0, boxes.size());
  }

  bool accepted_by_all(const Vec3& com, double edge_length, double theta) const {
    return nodes_.empty() || accepted(0, com, edge_length, theta);
  }

 private:
  struct Node {
    BBox box;
    std::int32_t left = -1, right = -1;  // -1: a single branch's box
  };
  std::vector<Node> nodes_;

  std::int32_t build(const std::vector<BBox>& boxes, std::size_t lo, std::size_t hi) {
    auto i = static_cast<std::int32_t>(nodes_.size());
    nodes_.emplace_back();
    if (hi - lo == 1) {
      nodes_[i].box = boxes[lo];
      return i;
    }
    std::size_t mid = lo + (hi - lo) / 2;
    std::int32_t left = build(boxes, lo, mid), right = build(boxes, mid, hi);
    BBox box = nodes_[left].box;
    box.add(nodes_[right].box.lo.x, nodes_[right].box.lo.y, nodes_[right].box.lo.z);
    box.add(nodes_[right].box.hi.x, nodes_[right].box.hi.y, nodes_[right].box.hi.z);
    nodes_[i] = {box, left, right};
    return i;
  }

  bool accepted(std::int32_t i, const Vec3& com, double edge_length, double theta) const {
    const Node& n = nodes_[i];
    if (mac_accept_bbox(n.box, com, edge_length, theta)) return true;
    if (n.left < 0) return false;
    return accepted(n.left, com, edge_length, theta) && accepted(n.right, com, edge_length, theta);
  }
};

using DistClock = std::chrono::steady_clock;
inline double ms_since(DistClock::time_point t) {
  return std::chrono::duration<double, std::milli>(DistClock::now() - t).count();
}

// Splitters of the global key order such that rank r owns keys in
// [split[r], split[r + 1]) and ~N/P particles: from kSamples regular samples
// of every rank's sorted keys, each weighted by the share of its rank's
// particles it stands for. Every rank computes the same result.
inline std::vector<std::uint64_t> choose_splitters(const std::vector<std::vector<std::uint64_t>>& samples,
                                                   const std::vector<RankBox>& boxes) {
  const int nranks = static_cast<int>(samples.size());
  struct Weighted {
    std::uint64_t key;
    double weight;
  };
  std::vector<Weighted> all;
  double total = 0.0;
  for (int r = 0; r < nranks; ++r) {
    if (samples[r].empty()) continue;
    double w = static_cast<double>(boxes[r].count) / static_cast<double>(samples[r].size());
    for (std::uint64_t k : samples[r]) all.push_back({k, w});
    total += static_cast<double>(boxes[r].count);
  }
  std::sort(all.begin(), all.end(), [](const Weighted& a, const Weighted& b) { return a.key < b.key; });
  std::vector<std::uint64_t> split(nranks + 1);
  split[0] = 0;
  split[nranks] = kKeyEnd;
  double cumulative = 0.0;
  std::size_t s = 0;
  for (int r = 1; r < nranks; ++r) {
    double target = total * r / nranks;
    while (s < all.size() && cumulative + all[s].weight <= target) cumulative += all[s++].weight;
    split[r] = std::max(split[r - 1], s < all.size() ? all[s].key : kKeyEnd);
  }
  return split;
}

inline BranchRecord branch_record(const FlatTree& f, NodeIndex i, std::uint64_t key, int owner) {
  const WalkNode& n = f.nodes[i];
  BranchRecord b;
  b.key = cell_first(key, n.depth);
  b.com = n.com;
  b.mass = n.mass;
  b.iab = f.iab[i];
  b.count = n.particle_count;
  b.local = i;
  b.owner = owner;
  b.depth = n.depth;
  b.leaf = n.leaf;
  return b;
}

// Appends the top tree over `branches` (sorted by key, disjoint cells) to
// `f` and returns its root; top_of_branch[b] is the node standing for
// branches[b]. Interior nodes get their multipoles from their children.
inline NodeIndex build_top_tree(FlatTree& f, const std::vector<BranchRecord>& branches, int me,
                                std::vector<NodeIndex>& top_of_branch) {
  auto grow = [&](std::size_t size) {
    f.nodes.resize(size);
    f.center.resize(size);
    f.iab.resize(size);
  };
  top_of_branch.assign(branches.size(), 0);
  const auto root = static_cast<NodeIndex>(f.nodes.size());
  grow(root + 1);

  // fills node i for the cell (depth, center) holding branches [lo, hi)
  auto fill = [&](auto&& self, NodeIndex i, std::size_t lo, std::size_t hi, int depth, Vec3 center) -> void {
    const double half_width = f.domain.half_width / static_cast<double>(std::uint64_t(1) << depth);
    if (hi - lo == 1 && branches[lo].depth == depth) {
      const BranchRecord& b = branches[lo];
      top_of_branch[lo] = i;
      if (b.owner == me) {
        // a copy of the local branch, leading into its local subtree
        f.nodes[i] = f.nodes[b.local];
        f.iab[i] = f.iab[b.local];
      } else {
        // a placeholder until step 5 grafts what this rank needs below it
        WalkNode w{};
        w.com = b.com;
        w.mass = b.mass;
        w.half_width = half_width;
        w.particle_count = 0;
        w.depth = b.depth;
        w.leaf = b.leaf;
        f.nodes[i] = w;
        f.iab[i] = b.iab;
      }
      f.center[i] = center;
      return;
    }
    // interior: one child per octant holding branches
    struct Range {
      unsigned octant;
      std::size_t lo, hi;
    };
    Range r[8];
    int k = 0;
    for (std::size_t j = lo; j < hi;) {
      unsigned oct = octant_digit(branches[j].key, depth + 1);
      std::size_t e = j;
      while (e < hi && octant_digit(branches[e].key, depth + 1) == oct) ++e;
      r[k++] = {oct, j, e};
      j = e;
    }
    auto first = static_cast<NodeIndex>(f.nodes.size());
    grow(first + k);
    std::uint32_t count = 0;
    for (int c = 0; c < k; ++c) {
      self(self, first + c, r[c].lo, r[c].hi, depth + 1, child_center(center, half_width, r[c].octant));
      count += f.nodes[first + c].particle_count;
    }
    WalkNode w{};
    w.half_width = half_width;
    w.first_child = first;
    w.num_children = static_cast<std::uint8_t>(k);
    w.particle_count = count;
    w.depth = static_cast<std::uint8_t>(depth);
    f.nodes[i] = w;
    f.center[i] = center;
    finalize_node(f, i);
  };
  fill(fill, root, 0, branches.size(), 0, f.domain.center);
  return root;
}

// What rank q (its branches' boxes in `boxes`) needs of this rank's branch at node b,
// appended to `nodes`/`particles` (see step 5 above); nothing if q accepts
// the branch as a whole.
inline void essential_subtree(const FlatTree& f, NodeIndex b, std::uint32_t branch, const BoxHierarchy& boxes,
                              double theta, std::vector<LetNode>& nodes, std::vector<LetParticle>& particles) {
  auto accepted = [&](NodeIndex i) {
    return boxes.accepted_by_all(f.nodes[i].com, 2.0 * f.nodes[i].half_width, theta);
  };
  if (accepted(b)) return;
  auto record = [&](NodeIndex i) {
    const WalkNode& n = f.nodes[i];
    LetNode r;
    r.com = n.com;
    r.mass = n.mass;
    r.half_width = n.half_width;
    r.iab = f.iab[i];
    r.leaf = n.leaf;
    return r;
  };
  // breadth-first, so the children of a node are consecutive
  std::vector<std::pair<NodeIndex, std::uint32_t>> queue;  // (node, its record)
  auto root_record = static_cast<std::uint32_t>(nodes.size());
  nodes.push_back(record(b));
  nodes.back().branch = branch;
  queue.push_back({b, root_record});
  for (std::size_t q = 0; q < queue.size(); ++q) {
    auto [i, r] = queue[q];
    const WalkNode& n = f.nodes[i];
    if (n.leaf) {
      nodes[r].particle_begin = static_cast<std::uint32_t>(particles.size());
      nodes[r].particle_count = n.particle_count;
      for (std::uint32_t p = n.particle_begin; p < n.particle_begin + n.particle_count; ++p) {
        particles.push_back({f.x[p], f.y[p], f.z[p], f.m[p]});
      }
      continue;
    }
    nodes[r].first_child = static_cast<std::uint32_t>(nodes.size());
    nodes[r].num_children = n.num_children;
    for (std::uint8_t c = 0; c < n.num_children; ++c) {
      NodeIndex ci = n.first_child + c;
      auto cr = static_cast<std::uint32_t>(nodes.size());
      nodes.push_back(record(ci));
      // accepted children stay childless (q only ever uses their multipole)
      if (!accepted(ci)) queue.push_back({ci, cr});
    }
  }
}

}  // namespace detail

// Collective. Repartitions `local` (this rank's particles, any order) by
// key range and builds this rank's locally essential tree into f (see the
// top of this file): f.particles holds this rank's particles afterwards,
// f.groups its target groups, and walks start at f.root. `local` is
// clobbered.
inline void build_distributed_tree(FlatTree& f, BuildScratch& ws, std::vector<Particle>& local, unsigned max_leaf,
                                   std::uint32_t group_size, double theta, DistributedBuildTiming* timing = nullptr) {
  using detail::DistClock;
  const int nranks = num_ranks(), me = this_rank();
  double exchange_ms = 0.0;
  auto start = DistClock::now();
  auto timed_exchange = [&exchange_ms](auto&& f) {
    auto t = DistClock::now();
    auto result = f();
    exchange_ms += detail::ms_since(t);
    return result;
  };

  // 1. global domain
  RankBox mine;
  std::tie(mine.lo, mine.hi) = particle_bounds(local);
  mine.count = local.size();
  auto boxes = timed_exchange([&] {
    auto g = allgather<RankBox>(std::span<const RankBox>(&mine, 1));
    std::vector<RankBox> b;
    for (auto& v : g) b.push_back(v[0]);
    return b;
  });
  Vec3 lo = boxes[0].lo, hi = boxes[0].hi;
  for (const auto& b : boxes) {
    lo = {std::min(lo.x, b.lo.x), std::min(lo.y, b.lo.y), std::min(lo.z, b.lo.z)};
    hi = {std::max(hi.x, b.hi.x), std::max(hi.y, b.hi.y), std::max(hi.z, b.hi.z)};
  }
  const Domain domain = domain_from_bounds(lo, hi);

  // 2. key ranges, and every particle to its owner. The particles arrive
  //    in the previous step's Morton order (or in input order), so regular
  //    samples of their keys are representative without sorting first;
  //    they are bucketed by owner, exchanged, and sorted once.
  const std::size_t n = local.size();
  const auto chunks = detail::num_sort_chunks(n);
  auto chunk_begin = [n, chunks](std::size_t c) { return n * c / chunks; };
  auto& keys = ws.route_keys;
  reset(keys, n);
  ttg_parallel_for(chunks, [&](std::uint32_t c) {
    for (std::size_t i = chunk_begin(c); i < chunk_begin(c + 1); ++i) keys[i] = morton_key(domain, local[i].pos);
  });
  constexpr std::size_t kSamples = 1024;
  std::vector<std::uint64_t> samples;
  for (std::size_t s = 0; s < kSamples && n > 0; ++s) samples.push_back(keys[n * s / kSamples]);
  auto all_samples = timed_exchange([&] { return allgather<std::uint64_t>(samples); });
  const std::vector<std::uint64_t> split = detail::choose_splitters(all_samples, boxes);
  auto owner = [&split](std::uint64_t key) {
    return static_cast<int>(std::upper_bound(split.begin() + 1, split.end() - 1, key) - (split.begin() + 1));
  };
  // count per (chunk, owner), then scatter into owner-contiguous runs
  std::vector<std::size_t> offset(std::size_t(chunks) * nranks, 0);
  ttg_parallel_for(chunks, [&](std::uint32_t c) {
    std::size_t* counts = &offset[std::size_t(c) * nranks];
    for (std::size_t i = chunk_begin(c); i < chunk_begin(c + 1); ++i) ++counts[owner(keys[i])];
  });
  std::vector<std::size_t> owner_begin(nranks + 1, 0);
  std::size_t pos = 0;
  for (int r = 0; r < nranks; ++r) {
    owner_begin[r] = pos;
    for (std::uint32_t c = 0; c < chunks; ++c) {
      std::size_t count = offset[std::size_t(c) * nranks + r];
      offset[std::size_t(c) * nranks + r] = pos;
      pos += count;
    }
    owner_begin[r + 1] = pos;
  }
  auto& routed = ws.routed;
  reset(routed, n);
  ttg_parallel_for(chunks, [&](std::uint32_t c) {
    std::size_t* next = &offset[std::size_t(c) * nranks];
    for (std::size_t i = chunk_begin(c); i < chunk_begin(c + 1); ++i) routed[next[owner(keys[i])]++] = local[i];
  });
  std::vector<std::span<const Particle>> out(nranks);
  for (int r = 0; r < nranks; ++r) {
    if (r != me) out[r] = std::span<const Particle>(routed.data() + owner_begin[r], owner_begin[r + 1] - owner_begin[r]);
  }
  auto received = timed_exchange([&] { return exchange<Particle>(out); });
  local.assign(routed.begin() + owner_begin[me], routed.begin() + owner_begin[me + 1]);
  for (int r = 0; r < nranks; ++r) local.insert(local.end(), received[r].begin(), received[r].end());
  sort_particles(f, ws, local, domain);

  // 3. the local tree, and its branches. Groups never start above a
  //    branch: a group rooted at a shared cell is replaced by the branches
  //    below it, so every box a walk tests lies within one branch (which
  //    the LET of step 5 relies on).
  const LeafRule rule{max_leaf, split[me], split[me + 1]};
  build_nodes(f, ws, rule, group_size);
  std::vector<BranchRecord> my_branches;
  if (!f.nodes.empty()) {
    std::vector<char> is_group(f.nodes.size(), 0);
    for (NodeIndex g : f.groups) is_group[g] = 1;
    std::vector<NodeIndex> groups;
    std::vector<std::pair<NodeIndex, bool>> stack{{0, false}};  // (node, below a shared group root)
    while (!stack.empty()) {
      auto [i, in_group] = stack.back();
      stack.pop_back();
      const WalkNode& node = f.nodes[i];
      std::uint64_t key = ws.keys[node.particle_begin];
      if (rule.owned(key, node.depth)) {
        my_branches.push_back(detail::branch_record(f, i, key, me));
        if (in_group) groups.push_back(i);
        continue;
      }
      in_group = in_group || is_group[i];
      for (int c = node.num_children; c-- > 0;) stack.push_back({node.first_child + c, in_group});  // keeps key order
    }
    for (NodeIndex g : f.groups) {
      if (rule.owned(ws.keys[f.nodes[g].particle_begin], f.nodes[g].depth)) groups.push_back(g);
    }
    f.groups.swap(groups);
  }

  // 4. the top tree over everyone's branches (with the boxes of their
  //    particles, for step 5); ranks' ranges are in key order, and each
  //    rank's branches are, so the concatenation is too
  ttg_parallel_for(static_cast<std::uint32_t>(my_branches.size()), [&](std::uint32_t b) {
    const WalkNode& node = f.nodes[my_branches[b].local];
    BBox box;
    for (std::uint32_t p = node.particle_begin; p < node.particle_begin + node.particle_count; ++p) {
      box.add(f.x[p], f.y[p], f.z[p]);
    }
    my_branches[b].lo = box.lo;
    my_branches[b].hi = box.hi;
  });
  auto gathered = timed_exchange([&] { return allgather<BranchRecord>(my_branches); });
  std::vector<BranchRecord> branches;
  std::vector<std::uint32_t> first_branch_of(nranks + 1, 0);  // rank r's: [first_branch_of[r], [r + 1])
  for (int r = 0; r < nranks; ++r) {
    branches.insert(branches.end(), gathered[r].begin(), gathered[r].end());
    first_branch_of[r + 1] = static_cast<std::uint32_t>(branches.size());
  }
  std::vector<NodeIndex> top_of_branch;
  f.root = detail::build_top_tree(f, branches, me, top_of_branch);

  // 5. locally essential tree
  std::vector<std::vector<LetNode>> let_nodes(nranks);
  std::vector<std::vector<LetParticle>> let_particles(nranks);
  ttg_parallel_for(static_cast<std::uint32_t>(nranks), [&](std::uint32_t q) {
    if (static_cast<int>(q) == me || first_branch_of[q] == first_branch_of[q + 1]) return;
    std::vector<BBox> boxes_q;
    for (std::uint32_t b = first_branch_of[q]; b < first_branch_of[q + 1]; ++b) {
      BBox box;
      box.lo = branches[b].lo;
      box.hi = branches[b].hi;
      boxes_q.push_back(box);
    }
    const detail::BoxHierarchy hierarchy(boxes_q);
    for (std::uint32_t b = first_branch_of[me]; b < first_branch_of[me + 1]; ++b) {
      detail::essential_subtree(f, branches[b].local, b, hierarchy, theta, let_nodes[q], let_particles[q]);
    }
  });
  auto in_nodes = timed_exchange([&] { return exchange<LetNode>(let_nodes); });
  auto in_particles = timed_exchange([&] { return exchange<LetParticle>(let_particles); });
  for (int s = 0; s < nranks; ++s) {
    if (s == me) continue;
    const auto node_base = static_cast<NodeIndex>(f.nodes.size());
    const auto particle_base = static_cast<std::uint32_t>(f.x.size());
    std::size_t np = particle_base + in_particles[s].size();
    f.x.resize(np);
    f.y.resize(np);
    f.z.resize(np);
    f.m.resize(np);
    for (std::size_t p = 0; p < in_particles[s].size(); ++p) {
      const LetParticle& lp = in_particles[s][p];
      f.x[particle_base + p] = lp.x;
      f.y[particle_base + p] = lp.y;
      f.z[particle_base + p] = lp.z;
      f.m[particle_base + p] = lp.m;
    }
    std::size_t nn = node_base + in_nodes[s].size();
    f.nodes.resize(nn);
    f.center.resize(nn);
    f.iab.resize(nn);
    for (std::size_t k = 0; k < in_nodes[s].size(); ++k) {
      const LetNode& r = in_nodes[s][k];
      WalkNode w{};
      w.com = r.com;
      w.mass = r.mass;
      w.half_width = r.half_width;
      w.first_child = node_base + r.first_child;
      w.num_children = r.num_children;
      w.leaf = r.leaf;
      w.particle_begin = particle_base + r.particle_begin;
      w.particle_count = r.particle_count;
      NodeIndex i = node_base + static_cast<NodeIndex>(k);
      f.nodes[i] = w;
      f.center[i] = r.com;  // only targets use centers, and these are sources
      f.iab[i] = r.iab;
      if (r.branch != LetNode::kNoBranch) {
        // the branch's placeholder in the top tree takes over its children
        WalkNode& t = f.nodes[top_of_branch[r.branch]];
        t.first_child = w.first_child;
        t.num_children = w.num_children;
        t.particle_begin = w.particle_begin;
        t.particle_count = w.particle_count;
      }
    }
  }

  if (timing != nullptr) {
    timing->exchange_ms = exchange_ms;
    timing->build_ms = detail::ms_since(start) - exchange_ms;
  }
}

// Collective: every rank's particles, on rank 0 (empty elsewhere) - for
// output.
inline std::vector<Particle> gather_particles(const std::vector<Particle>& local) {
  std::vector<std::span<const Particle>> out(num_ranks());
  out[0] = local;
  auto in = exchange<Particle>(out);
  std::vector<Particle> all;
  for (const auto& part : in) all.insert(all.end(), part.begin(), part.end());
  return all;
}

}  // namespace bh
