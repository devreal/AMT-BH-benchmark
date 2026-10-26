#pragma once

#include "key.h"
#include "particle.h"
#include "types.h"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace bh {

// The simulation domain is a cube; per-level edge length is derived by
// halving.
struct Domain {
  Vec3 center;
  double half_width = 0.0;
};

// The bounding cube of the axis-aligned box [lo, hi], padded slightly so
// particles exactly on its boundary are still inside.
inline Domain domain_from_bounds(const Vec3& lo, const Vec3& hi) {
  Domain domain;
  domain.center = Vec3{0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
  domain.half_width = 0.5 * std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z});
  if (!(domain.half_width > 0.0)) domain.half_width = 1.0;  // degenerate input (0 or 1 distinct position)
  domain.half_width *= 1.0 + 1e-9;  // avoid excluding particles exactly on the boundary
  return domain;
}

// The tight bounding cube of `particles`. Factored out of Tree's
// constructor (tree.cc) so a distributed build can compute ONE global
// Domain from every rank's particles combined - before any rank's local
// Tree is built - rather than each rank deriving its own (different, and
// therefore incompatible - see morton.h/distributed_upward.h) Domain from
// only the particles it happens to own.
inline Domain compute_domain(const std::vector<Particle>& particles) {
  Vec3 lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
          std::numeric_limits<double>::max()};
  Vec3 hi{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
          std::numeric_limits<double>::lowest()};
  for (const auto& p : particles) {
    lo.x = std::min(lo.x, p.pos.x);
    lo.y = std::min(lo.y, p.pos.y);
    lo.z = std::min(lo.z, p.pos.z);
    hi.x = std::max(hi.x, p.pos.x);
    hi.y = std::max(hi.y, p.pos.y);
    hi.z = std::max(hi.z, p.pos.z);
  }
  return domain_from_bounds(lo, hi);
}

// A node's box geometry (center, half-width) is fully determined by its Key
// (level + per-axis translation) and the domain's root box - no per-node
// storage or lookup is ever needed for it, so tasks that only need geometry
// never have to touch the tree at all, just this (tiny, trivially
// replicable) Domain plus whatever Key they already have.
inline std::pair<Vec3, double> box_geometry(const Domain& domain, const Key& key) {
  double cell_width = 2.0 * domain.half_width / static_cast<double>(1u << key.level());
  double half_width = 0.5 * cell_width;
  const auto& t = key.translation();
  Vec3 center{domain.center.x - domain.half_width + cell_width * (t[0] + 0.5),
              domain.center.y - domain.half_width + cell_width * (t[1] + 0.5),
              domain.center.z - domain.half_width + cell_width * (t[2] + 0.5)};
  return {center, half_width};
}

}  // namespace bh
