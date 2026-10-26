#include "tree.h"

#include <algorithm>
#include <array>

namespace bh {

namespace {

// Bit b of the octant index is set when the particle lies on the "high"
// side of `center` along axis b (x=0, y=1, z=2) - the same bit convention
// Key::child_at() (key.h) uses, so `key.child_at(octant_index(...))` gives
// the correct child key for the octant a particle actually falls into.
unsigned octant_index(const Vec3& pos, const Vec3& center) {
  unsigned idx = 0;
  if (pos.x >= center.x) idx |= 1u;
  if (pos.y >= center.y) idx |= 2u;
  if (pos.z >= center.z) idx |= 4u;
  return idx;
}

// Key::child_at() asserts level < Key::MAX_LEVEL, so recursion cannot go any
// deeper than that regardless - cap here at the same bound (rather than some
// larger guard) so the guard against infinite recursion on (near-)duplicate
// positions is the one that actually fires first.
constexpr int kMaxDepth = static_cast<int>(MAX_LEVEL);

}  // namespace

Tree::Tree(std::vector<Particle> particles, unsigned max_particles_per_leaf)
    : particles_(std::move(particles)) {
  domain_ = compute_domain(particles_);
  scratch_.resize(particles_.size());
  root_ = Key(0, {0, 0, 0});
  build_recursive(0, static_cast<std::uint32_t>(particles_.size()), root_, max_particles_per_leaf, 0);
}

Tree::Tree(std::vector<Particle> particles, unsigned max_particles_per_leaf, const Domain& domain)
    : particles_(std::move(particles)), domain_(domain) {
  scratch_.resize(particles_.size());
  root_ = Key(0, {0, 0, 0});
  build_recursive(0, static_cast<std::uint32_t>(particles_.size()), root_, max_particles_per_leaf, 0);
}

// Node geometry (center/half-width) is a pure function of `key` and
// `domain_` (see box_geometry() in domain.h) - computed on the fly here
// rather than threaded through the recursion or cached, so a child's box is
// derived from its OWN key rather than by halving its parent's stored box.
void Tree::build_recursive(std::uint32_t begin, std::uint32_t end, const Key& key,
                            unsigned max_particles_per_leaf, int depth) {
  std::uint32_t count = end - begin;

  if (count <= max_particles_per_leaf || depth >= kMaxDepth) {
    Node leaf;
    leaf.key = key;
    leaf.leaf = true;
    leaf.particle_begin = begin;
    leaf.particle_count = count;
    nodes_.emplace(key, std::move(leaf));
    return;
  }

  auto [center, half_width] = box_geometry(key);
  (void)half_width;

  // Counting-sort partition into up to 8 octants using `scratch_` (sized
  // once, in the constructor, for the whole build) instead of eight
  // per-call std::vectors - the original approach's repeated
  // allocation/reallocation as each bucket grew via push_back dominated
  // build cost far more than the O(N log N) partitioning itself.
  std::array<std::uint32_t, 8> counts{};
  for (std::uint32_t i = begin; i < end; ++i) {
    ++counts[octant_index(particles_[i].pos, center)];
  }

  std::array<std::uint32_t, 8> range_begin{};
  std::uint32_t running = begin;
  for (unsigned oct = 0; oct < 8; ++oct) {
    range_begin[oct] = running;
    running += counts[oct];
  }

  std::array<std::uint32_t, 8> cursor = range_begin;
  for (std::uint32_t i = begin; i < end; ++i) {
    unsigned oct = octant_index(particles_[i].pos, center);
    scratch_[cursor[oct]++] = particles_[i];
  }
  std::copy(scratch_.begin() + begin, scratch_.begin() + end, particles_.begin() + begin);

  Node interior;
  interior.key = key;
  interior.leaf = false;
  for (unsigned oct = 0; oct < 8; ++oct) {
    if (counts[oct] == 0) continue;
    interior.has_child[oct] = true;
    build_recursive(range_begin[oct], range_begin[oct] + counts[oct], key.child_at(oct),
                     max_particles_per_leaf, depth + 1);
  }
  nodes_.emplace(key, std::move(interior));
}

}  // namespace bh
