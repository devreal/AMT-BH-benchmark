#pragma once

#include "exchange_ttg.h"
#include "particle.h"

#include <ttg.h>

#include <cstdint>
#include <span>

// Multi-rank execution of the per-node pass, with replicated data: every
// rank holds all particles and builds the same tree, every task stays on
// the rank that creates it (local_keymap), and each rank only starts the
// force work it owns - an equal share of the tree-ordered particle array
// (owner_of). After the force pass, share_particles hands each rank's
// updated share to all others, so every rank enters the next step with all
// particles again. This splits the walk across ranks, but the tree build
// is repeated on every rank and every rank receives all particles every
// step. The grouped pass decomposes the domain instead
// (distributed_tree.h); the tree build's ttg_parallel_for uses
// local_keymap in both.

namespace bh {

// Keymap that keeps every task of a TT on the rank that created it (each
// rank runs its own instances of the graph's tasks; nothing is routed).
inline auto local_keymap() {
  int me = this_rank();
  return [me](const auto&) { return me; };
}

// The rank owning position `pos` of `n` items split into equal shares.
inline int owner_of(std::uint64_t pos, std::uint64_t n, int nranks) {
  return static_cast<int>(pos * static_cast<std::uint64_t>(nranks) / n);
}

// Collective: sends particles mine[0, count) - starting at `begin` - to
// every other rank, where place(begin, particles, count) stores them.
// Returns once every rank has placed every other rank's share.
template <typename Place>
void share_particles(const Particle* mine, std::uint64_t count, std::uint64_t begin, Place&& place) {
  const int nranks = num_ranks(), me = this_rank();
  if (nranks == 1) return;
  auto begins = allgather<std::uint64_t>(std::span<const std::uint64_t>(&begin, 1));
  auto shares = allgather<Particle>(std::span<const Particle>(mine, count));
  for (int r = 0; r < nranks; ++r) {
    if (r != me) place(begins[r][0], shares[r].data(), shares[r].size());
  }
}

}  // namespace bh
