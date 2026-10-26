#pragma once

#include "key.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bh {

// A reference depth used only for comparing/partitioning Keys along the
// Morton (Z-order) curve - e.g. for assigning contiguous ranges of the
// global order to different ranks (see distributed_upward.h). Capped well
// below Key::MAX_LEVEL (31) so the resulting code fits in 64 bits
// (NDIM * kSortDepth = 60, leaving headroom); a tree actually built deeper
// than this isn't representable by these functions (would need a wider
// code), but in practice that's far beyond what max_particles_per_leaf
// makes necessary for any realistic particle count.
constexpr Level kSortDepth = 20;
using MortonCode = std::uint64_t;

// The first Morton code (at kSortDepth) that any leaf under `key`'s box
// could ever occupy - i.e. `key` left-justified to the reference depth, so
// a shallow key and a deep key are directly comparable. Together with
// morton_span_width(), this is what makes "does this node's box extend
// outside my rank's range" a cheap arithmetic test rather than a tree walk:
// the defining property of a Morton order that makes contiguous-range
// domain decomposition work is that every node's descendants form a
// CONTIGUOUS range in this order.
inline MortonCode morton_lo(const Key& key) {
  MortonCode code = 0;
  Level pad = kSortDepth - key.level();
  const auto& t = key.translation();
  for (Level bit = 0; bit < key.level(); ++bit) {
    for (Dimension d = 0; d < NDIM; ++d) {
      code |= (MortonCode((t[static_cast<std::size_t>(d)] >> bit) & 1) << ((bit + pad) * NDIM + static_cast<Level>(d)));
    }
  }
  return code;
}

// Number of (reference-depth) Morton codes `key`'s box spans - i.e. the
// size of [morton_lo(key), morton_lo(key) + morton_span_width(key)).
inline MortonCode morton_span_width(const Key& key) {
  return MortonCode(1) << (NDIM * (kSortDepth - key.level()));
}

// Whether `key`'s entire box lies within the contiguous Morton range
// [lo, hi) - i.e. whether a rank owning exactly that range of leaves already
// has everything it needs to finalize `key`'s multipole from its own
// particles alone, with no contribution from any other rank.
inline bool range_contains(const Key& key, MortonCode lo, MortonCode hi) {
  MortonCode klo = morton_lo(key);
  return lo <= klo && klo + morton_span_width(key) <= hi;
}

// Decomposes the contiguous Morton range [lo, hi) into the minimal set of
// maximal octree-aligned blocks - the Key-space analogue of decomposing an
// address range into CIDR blocks. Each block's full span fits inside the
// range with no partial overlap at either edge, so IF every one of them
// held at least one particle, these would be exactly the nodes a rank
// owning [lo, hi) finalizes entirely on its own. In practice most of a
// sparse, non-uniform distribution's key-space is empty, and an empty
// block's node is never built at all - so this does NOT, in general, equal
// the number of redirects a rank's real upward pass produces; see
// distributed_upward.h, which computes that count from the actual built
// tree instead. Mainly useful here as a standalone, testable statement of
// what "maximal locally-ownable block" means along this order.
inline std::vector<Key> decompose_range(MortonCode lo, MortonCode hi) {
  std::vector<Key> blocks;
  while (lo < hi) {
    // Grow the block as long as doubling (one more level up, i.e. x8 more
    // Morton codes) stays aligned to `lo` and still fits in what's left.
    Level levels_up = 0;
    while (levels_up < kSortDepth) {
      MortonCode next_size = MortonCode(1) << (NDIM * (levels_up + 1));
      bool aligned = (lo % next_size) == 0;
      bool fits = next_size <= (hi - lo);
      if (!aligned || !fits) break;
      ++levels_up;
    }
    MortonCode block_size = MortonCode(1) << (NDIM * levels_up);
    Level level = kSortDepth - levels_up;

    std::array<Translation, NDIM> t{};
    MortonCode shifted = lo >> (levels_up * NDIM);
    for (Level bit = 0; bit < level; ++bit) {
      for (Dimension d = 0; d < NDIM; ++d) {
        if ((shifted >> (bit * NDIM + static_cast<Level>(d))) & 1) {
          t[static_cast<std::size_t>(d)] |= (Translation(1) << bit);
        }
      }
    }
    blocks.push_back(Key(level, t));
    lo += block_size;
  }
  return blocks;
}

}  // namespace bh
