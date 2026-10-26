#pragma once

#include "flat_tree.h"
#include "multipole.h"
#include "particle.h"
#include "replicated.h"
#include "simd.h"
#include "traversal_ttg.h"
#include "types.h"

#include <ttg.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

namespace bh {

// Coarse-grained alternative to bh_pass.h: one TTG task per *target group*
// (the topmost subtrees holding at most `group_size` particles, see
// FlatTree::groups) instead of one task per (target leaf, source node)
// pair. That decouples the number of tasks from max_leaf - the same way the
// HPX and Julia-Dagger codes of the AMT-BH benchmark chunk their force
// loops over bodies rather than over tree nodes - so max_leaf goes back to
// being a kernel-tuning parameter and group_size becomes the
// task-granularity knob.
//
// The multipoles are already final when the tree comes out of
// build_flat_tree (flat_build_ttg.h, itself parallel over TTG tasks), so
// this pass is a single TT, group_walk. Within its group it resolves the
// flat source tree for every *target node* - the topmost nodes with at most
// `target_size` particles, or leaves - using the same bounding-box MAC as
// traversal_ttg.h, into an interaction list that a SIMD kernel (simd.h)
// then evaluates for each of the target's particles, and hands every
// particle and its acceleration to the sink callback (kick/drift, see
// integrate_ttg.h). With several ranks, each walks the groups of its own
// particles in its locally essential tree (distributed_tree.h).
//
// The targets of a group share the upper part of their walks
// (GroupWalker): each node of the group's subtree, from the group root down
// to the targets, resolves the sources it can on behalf of all targets
// below it - those accepted for its particles' bounding box (then accepted
// for every target inside it) and those rejected for every point of that
// box (then opened, or direct-summed, by every target inside it) - and
// hands the rest on to its children. That produces exactly the
// interaction list of a separate walk from the root per target, while the
// top of the source tree is visited once per group instead of once per
// target.
//
// With target_size <= max_leaf the target nodes are exactly the leaves (an
// interior node always holds more than max_leaf particles), so both paths
// make the same accept/expand decisions and agree up to summation order.
// A larger target_size shares one walk among the particles of several
// small leaves - as the HPX code's SIMD walk shares one traversal among a
// packet of consecutive bodies - which matters most at max_leaf 1, where
// otherwise every particle walks the tree on its own. The larger target
// box makes the MAC more conservative, so it can only add interactions
// (and accuracy), never drop any.

// Receives each particle (in place in FlatTree::particles, so it may be
// updated directly - the walks only read the SoA copies x/y/z/m) and its
// acceleration.
using UpdateFn = std::function<void(Particle&, const Vec3&)>;

namespace detail {

// Every source that one target node's walk resolved: the particles of the
// leaves it direct-sums and the monopoles of the nodes the MAC accepted,
// gathered structure-of-arrays into one list so a single vectorized loop
// (accumulate_list) evaluates them all - a source leaf's own particle
// ranges are short (avg. ~3 at max_leaf 10) and would leave most SIMD
// lanes idle.
// With quadrupoles, accepted nodes instead go to `cells` for the scalar
// accel_from_source path. Used as a stack by GroupWalker: the entries a
// node of the group's subtree resolves stay on it while its descendants'
// are pushed on top.
struct InteractionList {
  static constexpr std::size_t kBlock = 2 * simd::kWidth;  // accumulate_list's unrolled step
  std::vector<double> x, y, z, m;
  std::size_t n = 0;
  std::vector<NodeIndex> cells;

  // drops everything pushed after `cells` held `num_cells` entries and
  // the list held `num`
  void truncate(std::size_t num, std::size_t num_cells) {
    n = num;
    cells.resize(num_cells);
  }
  void reserve_more(std::size_t k) {
    std::size_t need = n + k + kBlock;  // + room for padding
    if (need > x.size()) {
      std::size_t cap = std::max(need, 2 * x.size());
      x.resize(cap);
      y.resize(cap);
      z.resize(cap);
      m.resize(cap);
    }
  }
  void push(double px, double py, double pz, double pm) {
    x[n] = px;
    y[n] = py;
    z[n] = pz;
    m[n] = pm;
    ++n;
  }
  // zero-mass entries up to a multiple of kBlock
  void pad() {
    for (std::size_t i = n; i % kBlock != 0; ++i) {
      x[i] = y[i] = z[i] = m[i] = 0.0;
    }
  }
};

// Accelerations (without the factor G) on the NT points (px, py, pz)[0 ..
// NT) from every entry of `list`, written to acc[0 .. NT), U SIMD vectors
// of sources per step: NT * U independent rsqrt dependency chains per step,
// and the NT points share each source load. With Mask, entries at r2 == 0 (the point
// itself, or a duplicate position) are masked out; without softening they
// would be 0 * inf. With eps2 > 0 no mask is needed: r2 >= eps2, and the
// point's own entry has dx = 0.
template <int NT, int U, bool Mask>
inline void accumulate_list(const InteractionList& list, const double* px, const double* py, const double* pz,
                            double eps2, Vec3* acc) {
  static_assert(InteractionList::kBlock % (U * simd::kWidth) == 0);
  using simd::vdouble;
  const vdouble veps2 = simd::broadcast(eps2), zero{};
  vdouble vpx[NT], vpy[NT], vpz[NT];
  for (int t = 0; t < NT; ++t) {
    vpx[t] = simd::broadcast(px[t]);
    vpy[t] = simd::broadcast(py[t]);
    vpz[t] = simd::broadcast(pz[t]);
  }
  vdouble ax[NT][U]{}, ay[NT][U]{}, az[NT][U]{};
  for (std::size_t j = 0; j < list.n; j += U * simd::kWidth) {
    for (int u = 0; u < U; ++u) {
      std::size_t k = j + u * simd::kWidth;
      vdouble sx = simd::load(&list.x[k]), sy = simd::load(&list.y[k]), sz = simd::load(&list.z[k]);
      vdouble sm = simd::load(&list.m[k]);
      for (int t = 0; t < NT; ++t) {
        vdouble dx = sx - vpx[t], dy = sy - vpy[t], dz = sz - vpz[t];
        vdouble r2 = dx * dx + dy * dy + dz * dz + veps2;
        vdouble inv_r = simd::rsqrt(r2);
        if constexpr (Mask) inv_r = simd::select(r2 > zero, inv_r);
        vdouble s = sm * inv_r * inv_r * inv_r;
        ax[t][u] += s * dx;
        ay[t][u] += s * dy;
        az[t][u] += s * dz;
      }
    }
  }
  for (int t = 0; t < NT; ++t) {
    vdouble sx{}, sy{}, sz{};
    for (int u = 0; u < U; ++u) {
      sx += ax[t][u];
      sy += ay[t][u];
      sz += az[t][u];
    }
    acc[t] = {simd::hsum(sx), simd::hsum(sy), simd::hsum(sz)};
  }
}

// evaluate_target for the targets [tb, te).
template <bool Mask>
inline void evaluate_list(const FlatTree& f, std::uint32_t tb, std::uint32_t te, double eps2,
                          const InteractionList& list, Vec3* acc) {
  // 2 targets x 2 source vectors measured fastest on Neoverse-N1 (1, 2 or
  // 4 targets x 1 or 2 vectors: up to 25% slower, mostly from register
  // shuffling of the accumulators); with 16 registers its 12 accumulators
  // would spill.
  constexpr int kTargets = 2, kUnroll = simd::kRegisters >= 32 ? 2 : 1;
  std::uint32_t t = tb;
  for (; t + kTargets <= te; t += kTargets) {
    accumulate_list<kTargets, kUnroll, Mask>(list, &f.x[t], &f.y[t], &f.z[t], eps2, &acc[t - tb]);
  }
  for (; t < te; ++t) accumulate_list<1, 2, Mask>(list, &f.x[t], &f.y[t], &f.z[t], eps2, &acc[t - tb]);
}

// Accelerations (without the factor G) on every particle of target node
// `target`, written to acc[0 .. particle_count).
inline void evaluate_target(const FlatTree& f, NodeIndex target, const ForceParams& fp, const InteractionList& list,
                            Vec3* acc) {
  std::uint32_t tb = f.nodes[target].particle_begin, te = tb + f.nodes[target].particle_count;
  if (fp.eps2 > 0.0) {
    evaluate_list<false>(f, tb, te, fp.eps2, list, acc);
  } else {
    evaluate_list<true>(f, tb, te, fp.eps2, list, acc);
  }
  for (NodeIndex s : list.cells) {
    const WalkNode& src = f.nodes[s];
    for (std::uint32_t t = tb; t < te; ++t) {
      acc[t - tb] += accel_from_source(Vec3{f.x[t], f.y[t], f.z[t]}, src.mass, src.com, f.iab[s], fp.eps2, true, 1.0);
    }
  }
}

// The walks of one group task (see the top of this file). visit(i, open)
// resolves, for node i of the group's subtree, the sources in `open` - the
// ones i's ancestors could not resolve for all of their targets - pushing
// what it resolves onto `list`, then recurses into i's children with what
// is left, or evaluates `list` if i is a target.
struct GroupWalker {
  FlatTree& f;  // non-const only for the particles handed to on_result
  const ForceParams& fp;
  std::uint32_t target_size;
  const UpdateFn& on_result;
  InteractionList list;
  std::vector<NodeIndex> work;
  std::vector<std::vector<NodeIndex>> open;  // per depth below the group root
  std::vector<Vec3> acc;
  std::uint64_t missing = 0;  // opened nodes without particles or children

  GroupWalker(FlatTree& f, const ForceParams& fp, std::uint32_t target_size, const UpdateFn& on_result)
      : f(f), fp(fp), target_size(target_size), on_result(on_result), open(kMortonDepth + 2) {}

  void walk_group(NodeIndex root) {
    open[0].assign(1, f.root);  // the source tree's root
    visit(root, 0);
  }

  void push_source(NodeIndex s) {
    const WalkNode& sn = f.nodes[s];
    if (fp.use_quadrupole) {
      list.cells.push_back(s);
    } else {
      list.reserve_more(1);
      list.push(sn.com.x, sn.com.y, sn.com.z, sn.mass);
    }
  }

  // Direct-sums leaf s, or opens interior node s.
  void open_source(NodeIndex s) {
    const WalkNode& sn = f.nodes[s];
    // only a locally essential tree's stand-ins for remote nodes lack both
    // (distributed_tree.h), and its walks must never open one
    missing += sn.leaf ? sn.particle_count == 0 : sn.num_children == 0;
    if (sn.leaf) {
      // includes the target's own leaves: the i == j term has dx = 0 and is
      // masked at r2 = 0 (eps = 0) or contributes 0 (eps > 0)
      list.reserve_more(sn.particle_count);
      for (std::uint32_t p = sn.particle_begin; p < sn.particle_begin + sn.particle_count; ++p) {
        list.push(f.x[p], f.y[p], f.z[p], f.m[p]);
      }
    } else {
      for (std::uint8_t c = 0; c < sn.num_children; ++c) work.push_back(sn.first_child + c);
    }
  }

  void visit(NodeIndex i, int depth) {
    const WalkNode& t = f.nodes[i];
    bool target = t.leaf || t.particle_count <= target_size;
    BBox box;
    for (std::uint32_t p = t.particle_begin; p < t.particle_begin + t.particle_count; ++p) box.add(f.x[p], f.y[p], f.z[p]);

    std::size_t list_mark = list.n, cells_mark = list.cells.size();
    std::vector<NodeIndex>& rest = open[depth + 1];  // only used if !target
    rest.clear();
    work.assign(open[depth].begin(), open[depth].end());
    while (!work.empty()) {
      NodeIndex s = work.back();
      work.pop_back();
      const WalkNode& sn = f.nodes[s];
      // Leaves are MAC-tested too (as in traversal_ttg.h): a distant leaf
      // is one monopole, not a direct sum over its particles.
      double edge = 2.0 * sn.half_width;
      if (mac_accept_bbox(box, sn.com, edge, fp.theta)) {
        push_source(s);
      } else if (target || mac_reject_all(box, sn.com, edge, fp.theta)) {
        open_source(s);
      } else {
        rest.push_back(s);
      }
    }

    if (target) {
      list.pad();
      acc.resize(t.particle_count);
      evaluate_target(f, i, fp, list, acc.data());
      for (std::uint32_t k = 0; k < t.particle_count; ++k) {
        on_result(f.particles[t.particle_begin + k], fp.G * acc[k]);
      }
    } else {
      for (std::uint8_t c = 0; c < t.num_children; ++c) visit(t.first_child + c, depth + 1);
    }
    list.truncate(list_mark, cells_mark);
  }
};

}  // namespace detail

// Runs the pass over all of flat's groups, on this rank; with several
// ranks, flat is this rank's locally essential tree (distributed_tree.h).
// Returns how many nodes the walks opened without having their particles or
// children - always 0 unless a locally essential tree is incomplete.
inline std::uint64_t run_grouped_pass(FlatTree& flat, const ForceParams& fp, std::uint32_t target_size,
                                      const UpdateFn& on_result) {
  std::atomic<std::uint64_t> missing = 0;
  const auto num_groups = static_cast<std::uint32_t>(flat.groups.size());
  ttg::Edge<std::uint32_t, void> spawn_edge, walk_edge;  // group index

  auto spawn_tt = ttg::make_tt(
      [num_groups](const std::uint32_t&) {
        for (std::uint32_t g = 0; g < num_groups; ++g) ttg::sendk<0>(g);
      },
      ttg::edges(spawn_edge), ttg::edges(walk_edge), "spawn_walks");

  auto group_walk_tt = ttg::make_tt(
      [&flat, &fp, target_size, &on_result, &missing](const std::uint32_t& g) {
        detail::GroupWalker walker(flat, fp, target_size, on_result);
        walker.walk_group(flat.groups[g]);
        missing += walker.missing;
      },
      ttg::edges(walk_edge), ttg::edges(), "group_walk");

  spawn_tt->set_keymap(local_keymap());
  group_walk_tt->set_keymap(local_keymap());
  ttg::make_graph_executable(spawn_tt.get(), group_walk_tt.get());
  spawn_tt->invoke(0);
  ttg::fence();
  return missing;
}

}  // namespace bh
