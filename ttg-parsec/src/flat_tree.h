#pragma once

#include "domain.h"
#include "multipole.h"
#include "particle.h"
#include "types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace bh {

using NodeIndex = std::uint32_t;

// Allocator whose resize() leaves elements uninitialized, for the large
// per-step arrays of the tree build that a parallel phase fully overwrites
// right afterwards: value-initializing them would be a serial pass over
// hundreds of MB (measured at ~45% of the build at 1M particles on 128
// threads), and leaving it out also moves the first touch of each page
// into the parallel phase. Only for implicit-lifetime types (trivially
// copyable, trivially destructible), so skipping construction is fine.
template <typename T>
struct UninitAllocator : std::allocator<T> {
  static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>);
  template <typename U>
  struct rebind {
    using other = UninitAllocator<U>;
  };
  UninitAllocator() = default;
  template <typename U>
  UninitAllocator(const UninitAllocator<U>&) noexcept {}
  template <typename U>
  void construct(U*) noexcept {}  // value-initialization (resize, vector(n)): skipped
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
  }
};

template <typename T>
using UninitVector = std::vector<T, UninitAllocator<T>>;

// Array-based octree for the grouped pass (grouped_ttg.h), built directly
// from Morton-sorted particles (flat_build_ttg.h) rather than via Tree's
// unordered_map<Key, Node> - walking hash-map nodes costs ~5x more than
// walking flat arrays (measured on the three-galaxy data set). Node 0 is
// the root and the children of a node are consecutive, starting at
// first_child. A node's particles are the contiguous range
// [particle_begin, +particle_count) of `particles` (Morton order), whether
// it is a leaf or not. Particle positions/masses are also kept
// structure-of-arrays so the direct-sum loops vectorize.
//
// The fields the walk touches for every visited node are packed into one
// 64-byte WalkNode; per-node data only needed for target leaves (box
// center) or the quadrupole term (iab) lives in separate arrays.
struct WalkNode {
  Vec3 com;
  double mass = 0.0;
  double half_width = 0.0;
  NodeIndex first_child = 0;
  std::uint32_t particle_begin = 0, particle_count = 0;
  std::uint8_t num_children = 0;
  std::uint8_t leaf = 0;
  std::uint8_t depth = 0;  // root = 0; only used while building
};

struct FlatTree {
  Domain domain;
  UninitVector<WalkNode> nodes;
  UninitVector<Vec3> center;
  UninitVector<Sym3> iab;
  // Roots of the target groups: the topmost nodes with at most group_size
  // particles (or leaves). Every leaf is in exactly one group.
  std::vector<NodeIndex> groups;

  // Morton order. A plain vector so it can trade buffers with the caller's
  // particle vector (integrate_ttg.h); it is only resized when the particle
  // count changes, so its value-initialization is paid once.
  std::vector<Particle> particles;
  UninitVector<double> x, y, z, m;

  // Where walks start: 0 for a tree of all particles; with a domain
  // decomposition, the top tree assembled behind the local nodes (see
  // distributed_tree.h), and x/y/z/m then continue with remote particles
  // past particles.size().
  NodeIndex root = 0;

  std::size_t num_nodes() const { return nodes.size(); }
};

// 63-bit Morton keys, 21 bits per axis: the digit selecting a node's child
// at depth d (root = depth 0) is bits [63 - 3d, 66 - 3d), with x in the low
// bit of the digit - the same octant numbering as Key::child_at() and
// tree.cc's octant_index(). Depth is therefore capped at 21 here, vs.
// Key::MAX_LEVEL for Tree; deeper nodes only arise from (near-)duplicate
// positions, which end up in one oversized leaf either way.
constexpr int kMortonDepth = 21;

inline std::uint64_t spread_bits3(std::uint32_t v) {
  std::uint64_t x = v & 0x1fffffu;
  x = (x | (x << 32)) & 0x1f00000000ffffull;
  x = (x | (x << 16)) & 0x1f0000ff0000ffull;
  x = (x | (x << 8)) & 0x100f00f00f00f00full;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
  x = (x | (x << 2)) & 0x1249249249249249ull;
  return x;
}

inline std::uint64_t morton_key(const Domain& d, const Vec3& p) {
  constexpr double cells = static_cast<double>(1u << kMortonDepth);
  double scale = cells / (2.0 * d.half_width);
  auto quantize = [&](double v, double lo) {
    double q = std::floor((v - lo) * scale);
    return static_cast<std::uint32_t>(std::clamp(q, 0.0, cells - 1.0));
  };
  std::uint32_t qx = quantize(p.x, d.center.x - d.half_width);
  std::uint32_t qy = quantize(p.y, d.center.y - d.half_width);
  std::uint32_t qz = quantize(p.z, d.center.z - d.half_width);
  return spread_bits3(qx) | (spread_bits3(qy) << 1) | (spread_bits3(qz) << 2);
}

inline unsigned octant_digit(std::uint64_t key, int depth) {
  return static_cast<unsigned>((key >> (63 - 3 * depth)) & 7u);
}

// One past the largest key.
constexpr std::uint64_t kKeyEnd = std::uint64_t(1) << 63;

// The keys of the cell at `depth` that holds `key`: [first, first + size).
inline std::uint64_t cell_size(int depth) { return std::uint64_t(1) << (63 - 3 * depth); }
inline std::uint64_t cell_first(std::uint64_t key, int depth) { return key & ~(cell_size(depth) - 1); }

// When the build stops splitting a node: at most max_leaf particles (or the
// depth cap) - but only once its cell lies entirely in the key range
// [key_lo, key_hi) this tree owns. With a domain decomposition, cells that
// reach into other ranks' ranges are always split, down to cells that are
// entirely local (at the depth cap a cell is a single key, so always is);
// with the whole range, this is just the max_leaf rule.
struct LeafRule {
  unsigned max_leaf = 1;
  std::uint64_t key_lo = 0, key_hi = kKeyEnd;

  bool owned(std::uint64_t key, int depth) const {
    std::uint64_t first = cell_first(key, depth);
    return first >= key_lo && first + cell_size(depth) <= key_hi;
  }
  // node of `count` particles from sorted keys[begin]
  bool stops(const std::uint64_t* keys, std::uint32_t begin, std::uint32_t count, int depth) const {
    return (count <= max_leaf || depth >= kMortonDepth) && owned(keys[begin], depth);
  }
};

namespace detail {

struct OctantRange {
  unsigned octant;
  std::uint32_t lo, hi;
};

// The non-empty child octants of sorted keys [lo, hi) at child depth
// `cdepth`, ascending; each is contiguous, so its end is a binary search.
inline int split_octants(const std::uint64_t* keys, std::uint32_t lo, std::uint32_t hi, int cdepth,
                         OctantRange out[8]) {
  int k = 0;
  std::uint32_t j = lo;
  while (j < hi) {
    unsigned oct = octant_digit(keys[j], cdepth);
    auto e = static_cast<std::uint32_t>(
        std::partition_point(keys + j, keys + hi, [&](std::uint64_t c) { return octant_digit(c, cdepth) <= oct; }) -
        keys);
    out[k++] = {oct, j, e};
    j = e;
  }
  return k;
}

inline Vec3 child_center(const Vec3& c, double half_width, unsigned oct) {
  double q = 0.5 * half_width;
  return {c.x + ((oct & 1u) ? q : -q), c.y + ((oct & 2u) ? q : -q), c.z + ((oct & 4u) ? q : -q)};
}

// Whether a node holding `count` particles, whose parent holds
// `parent_count` (the root: UINT32_MAX), roots a target group (see
// FlatTree::groups).
inline bool is_group_root(std::uint32_t count, bool leaf, std::uint32_t parent_count, std::uint32_t group_size) {
  return (leaf || count <= group_size) && parent_count > group_size;
}

// Sets up node i of `f` for particles [begin, begin + count): everything but
// its children (split_node) and its multipole (finalize_node).
inline void init_node(FlatTree& f, NodeIndex i, std::uint32_t begin, std::uint32_t count, const Vec3& center,
                      double half_width, int depth, const std::uint64_t* keys, const LeafRule& rule) {
  WalkNode& w = f.nodes[i];
  w = WalkNode{};
  w.particle_begin = begin;
  w.particle_count = count;
  w.half_width = half_width;
  w.depth = static_cast<std::uint8_t>(depth);
  w.leaf = rule.stops(keys, begin, count, depth);
  f.center[i] = center;
}

// Creates the children of interior node i at indices [first, first + k) of
// `f` (which must already hold them) and returns k; the group roots among
// them are appended to `groups`.
inline int split_node(FlatTree& f, NodeIndex i, NodeIndex first, const std::uint64_t* keys, const LeafRule& rule,
                      std::uint32_t group_size, std::vector<NodeIndex>& groups) {
  OctantRange r[8];
  WalkNode& n = f.nodes[i];
  int k = split_octants(keys, n.particle_begin, n.particle_begin + n.particle_count, n.depth + 1, r);
  n.first_child = first;
  n.num_children = static_cast<std::uint8_t>(k);
  for (int c = 0; c < k; ++c) {
    NodeIndex ci = first + static_cast<NodeIndex>(c);
    init_node(f, ci, r[c].lo, r[c].hi - r[c].lo, child_center(f.center[i], n.half_width, r[c].octant),
              0.5 * n.half_width, n.depth + 1, keys, rule);
    if (is_group_root(r[c].hi - r[c].lo, f.nodes[ci].leaf, n.particle_count, group_size)) groups.push_back(ci);
  }
  return k;
}

struct KeyRange {
  std::uint32_t lo, hi;
  int depth;
};

// The number of nodes (including its root) of the subtree that
// build_subtree creates over sorted keys [lo, hi) from depth `depth` - the
// same splits, without writing anything, so the subtrees can be laid out
// before they are built.
inline std::uint32_t count_subtree(const std::uint64_t* keys, std::uint32_t lo, std::uint32_t hi, int depth,
                                   const LeafRule& rule, std::vector<KeyRange>& stack) {
  std::uint32_t count = 0;
  stack.clear();
  stack.push_back({lo, hi, depth});
  while (!stack.empty()) {
    KeyRange kr = stack.back();
    stack.pop_back();
    ++count;
    if (rule.stops(keys, kr.lo, kr.hi - kr.lo, kr.depth)) continue;
    OctantRange r[8];
    int k = split_octants(keys, kr.lo, kr.hi, kr.depth + 1, r);
    for (int c = 0; c < k; ++c) stack.push_back({r[c].lo, r[c].hi, kr.depth + 1});
  }
  return count;
}

// Finalizes node i's multipole from its particles (leaf) or its children's
// already-final multipoles (interior), using the same raw-moment /
// parallel-axis machinery as upward_ttg.h so both passes produce the same
// moments.
inline void finalize_node(FlatTree& f, NodeIndex i) {
  WalkNode& n = f.nodes[i];
  const Vec3& ref = f.center[i];
  RawMoment rm;
  if (n.leaf) {
    std::uint32_t b = n.particle_begin, e = b + n.particle_count;
    for (std::uint32_t p = b; p < e; ++p) rm += raw_moment_of_particle(f.m[p], Vec3{f.x[p], f.y[p], f.z[p]}, ref);
  } else {
    for (std::uint8_t c = 0; c < n.num_children; ++c) {
      const WalkNode& ch = f.nodes[n.first_child + c];
      rm += raw_moment_from_multipole(Multipole{ch.mass, ch.com, f.iab[n.first_child + c]}, ref);
    }
  }
  Multipole mp = finalize_raw_moment(rm, ref);
  n.mass = mp.mass;
  n.com = mp.com;
  f.iab[i] = mp.iab;
}

// Builds the subtree below the (already initialized) node `root` in place:
// breadth-first, its other nodes at [base, base + count_subtree(...) - 1),
// children consecutive, then all of its multipoles bottom-up. Group roots
// below `root` are appended to `groups`.
inline void build_subtree(FlatTree& f, NodeIndex root, NodeIndex base, const std::uint64_t* keys, const LeafRule& rule,
                          std::uint32_t group_size, std::vector<NodeIndex>& groups) {
  NodeIndex next = base;
  if (!f.nodes[root].leaf) next += split_node(f, root, next, keys, rule, group_size, groups);
  for (NodeIndex i = base; i < next; ++i) {
    if (!f.nodes[i].leaf) next += split_node(f, i, next, keys, rule, group_size, groups);
  }
  for (NodeIndex i = next; i-- > base;) finalize_node(f, i);
  finalize_node(f, root);
}

}  // namespace detail

}  // namespace bh
