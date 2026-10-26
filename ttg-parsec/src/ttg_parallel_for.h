#pragma once

#include "replicated.h"

#include <ttg.h>

#include <cstdint>

namespace bh {

// Runs f(i) for every i in [0, n) as TTG tasks and fences, so host-side
// data-parallel phases (the tree build in flat_build_ttg.h) run on the
// runtime's own worker threads instead of a second thread pool competing
// with them for the same cores. A single spawner task fans the indices out
// so the tasks are scheduled by the runtime rather than started from the
// calling thread. With several ranks, every rank runs all of [0, n) on its
// own (replicated data, see replicated.h).
template <typename F>
void ttg_parallel_for(std::uint32_t n, F&& f) {
  if (n == 0) return;
  ttg::Edge<std::uint32_t, void> spawn_edge, body_edge;
  auto spawn_tt = ttg::make_tt(
      [n](const std::uint32_t&) {
        for (std::uint32_t i = 0; i < n; ++i) ttg::sendk<0>(i);
      },
      ttg::edges(spawn_edge), ttg::edges(body_edge), "parallel_for_spawn");
  auto body_tt = ttg::make_tt([&f](const std::uint32_t& i) { f(i); }, ttg::edges(body_edge), ttg::edges(),
                              "parallel_for_body");
  spawn_tt->set_keymap(local_keymap());
  body_tt->set_keymap(local_keymap());
  ttg::make_graph_executable(spawn_tt.get(), body_tt.get());
  spawn_tt->invoke(0);
  ttg::fence();
}

}  // namespace bh
