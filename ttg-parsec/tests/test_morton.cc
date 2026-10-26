#include "../src/morton.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <utility>

using namespace bh;

namespace {

bool check_range(MortonCode lo, MortonCode hi) {
  auto blocks = decompose_range(lo, hi);
  if (lo == hi) {
    if (!blocks.empty()) {
      std::cerr << "FAIL: empty range [" << lo << "," << hi << ") produced " << blocks.size()
                << " blocks\n";
      return false;
    }
    return true;
  }

  MortonCode cursor = lo;
  for (const Key& b : blocks) {
    if (morton_lo(b) != cursor) {
      std::cerr << "FAIL: block does not start where expected for [" << lo << "," << hi << ")\n";
      return false;
    }
    if (!range_contains(b, lo, hi)) {
      std::cerr << "FAIL: block not contained in [" << lo << "," << hi << ")\n";
      return false;
    }
    cursor += morton_span_width(b);
  }
  if (cursor != hi) {
    std::cerr << "FAIL: blocks don't exactly cover [" << lo << "," << hi << ") - ended at " << cursor
               << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;

  ok &= check_range(5, 5);  // empty
  ok &= check_range(7, 8);  // single leaf

  MortonCode full = MortonCode(1) << (NDIM * kSortDepth);
  ok &= check_range(0, full);  // the whole space

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<MortonCode> dist(0, full);
  for (int i = 0; i < 2000; ++i) {
    MortonCode a = dist(rng), b = dist(rng);
    if (a > b) std::swap(a, b);
    if (!check_range(a, b)) ok = false;
  }

  // The root's span should be the entire reference-depth space.
  Key root(0, {0, 0, 0});
  if (morton_lo(root) != 0 || morton_span_width(root) != full) {
    std::cerr << "FAIL: root's span should be [0, " << full << ")\n";
    ok = false;
  }

  // child_at() order should match Morton order: children tile the parent's
  // span exactly, in child_at(0..7) order - the same bit convention
  // tree.cc's octant_index()/build() relies on.
  Key parent(3, {2, 5, 1});
  MortonCode cursor = morton_lo(parent);
  for (int idx = 0; idx < Key::num_children(); ++idx) {
    Key child = parent.child_at(idx);
    if (morton_lo(child) != cursor) {
      std::cerr << "FAIL: child_at(" << idx << ") out of Morton order\n";
      ok = false;
    }
    cursor += morton_span_width(child);
  }
  if (cursor != morton_lo(parent) + morton_span_width(parent)) {
    std::cerr << "FAIL: children don't exactly tile the parent's span\n";
    ok = false;
  }

  if (ok) {
    std::cout << "PASS: Morton range decomposition/containment are correct\n";
    return 0;
  }
  return 1;
}
