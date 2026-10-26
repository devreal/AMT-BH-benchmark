# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Barnes-Hut gravitational N-body solver expressed as a task graph over
[TTG](https://github.com/TESSEorg/ttg) (Template Task Graph), on top of the
PaRSEC runtime. It exists primarily to measure TTG/PaRSEC per-task overhead
on a well-understood, physically checkable workload, and secondarily to
compare against [PEPC](https://github.com/SC-HPC-JSC/pepc)'s `pepc-gravity`
frontend as a reference implementation.

## Build / run / test

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

- Requires CMake ≥ 3.19 and a C++20 compiler (GCC/Clang: the force kernel
  uses vector extensions).
- Our own targets build with `-march=native` (`-DBH_NATIVE_ARCH=OFF` to
  disable); without it an x86 build falls back to the SSE2 kernel.
- Defaults to a `Release` build (this project measures per-task overhead,
  which an unoptimized build would distort) — override with
  `-DCMAKE_BUILD_TYPE=Debug` if needed.
- TTG (and its PaRSEC/MADNESS dependencies) are pulled in via `FetchContent`
  from a fork/branch pinned in `cmake/modules/FindOrFetchTTG.cmake`
  (`devreal/ttg.git`, branch `ttg-kernel-batching_aggregators` — it carries
  an aggregator fix not yet upstream; see the comment in that file). If you
  already have a TTG build elsewhere with the same deps cloned, point
  `-DFETCHCONTENT_BASE_DIR=/path/to/that/build/_deps` at it so configure
  reuses the clones instead of re-fetching from GitHub — but never hardcode
  such a path into `CMakeLists.txt`, since it won't exist on other systems.
- Run a single test: `ctest --test-dir build -R <name>` (e.g. `-R test_morton`).
  All tests run under `mpirun -n 1` (`SINGLERANKONLY`); multi-rank runs
  are checked by hand (see "Multi-rank execution" below).
- Run the solver directly:
  ```sh
  build/src/pepc-ttg-parsec --input data/two_body.csv --steps 100 --dt 1e-3
  ```
  Without `--input`, a random uniform-in-cube distribution is generated.
  The CSV reader (`src/io.cc`) memory-maps the file and parses it with one
  thread per usable CPU before `ttg::execute()`; a header line (as written
  by the AMT-BH data generator) is skipped. Particle ids must be
  0..N-1 (the per-node path indexes its output by id).
  See `src/main.cc` for the full flag list (`--theta`, `--eps`, `--max-leaf`,
  `--no-quadrupole`, `--group-size`, `--target-size`, `--energy-every`, `--time`, `--vtk-prefix`, `--n`,
  `--seed`, `--spread`, `--vel-spread`, ...).

## Architecture

Each timestep is one TTG graph, fenced once (not once per pass):

1. **Tree build** (sequential, host-side, `src/tree.h`/`src/tree.cc`):
   particles are recursively partitioned into an octree until each leaf
   holds at most `--max-leaf` particles. Nodes are addressed by `Key`
   (level + per-axis translation, `src/key.h`), not an allocated id, so
   parent/child relationships are pure arithmetic on the key
   (`Key::parent()`, `Key::child_at()`) and `Tree` stores nodes in a sparse
   `unordered_map<Key, Node>` rather than a dense array. Each `Node` stores
   an `std::array<bool, 8> has_child` bitmap (not a list of child keys —
   children are reconstructed on demand via `key.child_at(idx)`).

2. **Upward pass** (`src/upward_ttg.h`): one `seed_leaf` task per leaf
   finalizes that leaf's own multipole (deposited for the downward pass's
   MAC test of leaf sources) and forwards it as a raw moment about its
   *parent's* box center; a `combine_up`
   aggregator per interior node (target = that node's actual child count)
   fires once every child's moment has arrived, finalizes the node's
   multipole, and re-expresses/forwards it up to its own parent via the
   same edge (self-loop). The root's aggregator firing is itself the
   signal that every multipole is final — no host-side fence is needed
   before the downward pass starts.

3. **Downward pass** (`src/traversal_ttg.h`): a MAC (multipole acceptance
   criterion, `theta`) driven descent, batched per *target leaf* (keyed by
   `NN{target leaf, source node}`) rather than per particle — every particle
   in a leaf shares the same traversal decisions, cutting task count by
   roughly `max-leaf`. A source node (leaf or interior) is either accepted
   (multipole applied to every target particle), direct-summed (a leaf), or
   expanded into its children; results
   aggregate back up to a per-leaf sink.

4. **Integration**: leapfrog kick/drift is fused directly into the downward
   pass's per-leaf sink callback, so particles in a leaf that finishes early
   start integrating while other leaves are still mid-traversal
   (`src/integrate_ttg.h`, `src/integrate.h`).

**Two pipelines, selected by `--group-size`.** Steps 1-3 above are the
*per-node* path (`--group-size 0`): one task per (target leaf, source node)
pair, ~280 tasks/body at `--max-leaf 10`, so task count is set by leaf size.
The default is the *grouped* path (default `--group-size 256`), which never
builds a `Tree`:
- `src/flat_build_ttg.h` builds the array-based `FlatTree`
  (`src/flat_tree.h`) in parallel from Morton keys, as a sequence of
  `ttg_parallel_for` phases (`src/ttg_parallel_for.h`, one fence each):
  chunked bounding box, keys + per-chunk sort, sample sort into buckets, a
  serial breadth-first backbone down to subtrees of <= 4096 particles, one
  task per subtree that counts its nodes (to lay the subtrees out), then
  one task per subtree that builds it in place in the global arrays and
  computes its multipoles - there is no separate upward pass. Large
  arrays use `UninitVector` so they aren't zero-filled serially, and the
  `FlatTree` + `BuildScratch` live in a `GroupedWorkspace` (owned by
  `main.cc`) reused across steps; particles are kicked/drifted in place in
  the tree's Morton-ordered array, which becomes the next step's input -
  so the grouped `step_particles` returns particles in Morton order, not
  by id. Fresh per-step allocation cost >2 s/step at 32M bodies.
- `src/grouped_ttg.h` runs one `group_walk` task per group (topmost subtree
  of <= G particles). Inside it, each *target node* (topmost node with <=
  `--target-size` particles, default 32, or a leaf) gets an interaction
  list (leaf particles + accepted monopoles, structure-of-arrays) that a
  SIMD kernel (`src/simd.h`: native-width GCC vector extensions, rsqrt
  estimate + Newton steps per ISA) evaluates for each target particle. The
  lists are built by one walk per group shared among its targets
  (`GroupWalker`): every node of the group's subtree resolves the sources
  that are accepted, or rejected, for all of its particles at once and
  hands the rest down, which yields exactly the lists of separate per-target
  walks. The MAC (`mac_accept_bbox`, also used by the per-node path) tests
  against the bounding box of the target's particles, not its cell. With
  `--target-size` <= `--max-leaf` the targets are
  exactly the leaves and forces match the per-node path up to summation
  order (checked in `test_bh_vs_bruteforce`); larger target sizes share a
  walk among small leaves (like HPX's packet walk) with a more conservative
  MAC. Quadrupole terms still go through the scalar `accel_from_source`.
This mirrors how the HPX/Julia-Dagger codes of the AMT-BH benchmark chunk
their force loops over bodies, so leaf size is a kernel parameter and group
size is the granularity knob. Morton keys cap the grouped tree's depth at
21 (vs. `Key::MAX_LEVEL` for `Tree`).

`src/bh_pass.h` wires the upward pass's completion edge directly into the
downward pass's entry point, so both run as a single graph with one
`ttg::fence()` per step (see `run_bh_pass`), instead of fencing in between.

Two shared, mutex-protected stores bridge the passes without either pass
header depending on the other: `LeafParticleStore` (a leaf's own particles,
written once by `seed_leaf`, read by the downward pass) and
`MultipoleStore` (a node's finalized multipole, written once by
`combine_up`, read whenever a traversal needs to test a source node). Both
are written concurrently by many tasks (hence the mutex) but read only
after the pass that writes them has fully completed (hence no lock needed
on read).

See `Whitepaper_Barnes_Hut.pdf` for the underlying method and
`src/multipole.h` for the raw-moment/quadrupole math (including the
parallel-axis-theorem shift used when re-expressing a moment about a new
reference point as it moves up the tree).

### TTG usage conventions in this codebase

- Tasks are plain lambdas passed to `ttg::make_tt`; they do **not** take a
  trailing `outs` tuple parameter — TTG's thread-local-pointer mechanism
  supplies output terminals, so `ttg::send<i>`/`ttg::sendk<i>` are called
  directly from inside the lambda body.
- Any type sent over a `ttg::Edge` needs an explicit `serialize()` (MADNESS's
  archive framework, which TTG's PaRSEC backend always uses, does not
  auto-serialize these types) — see `Vec3`, `Key`, `Particle`, `NN`,
  `Contribution` for the pattern (always define both the unversioned and
  versioned overload).
- `ttg::make_aggregator`'s target-count function is typically captured by
  reference to the `Tree` (e.g. `tree.num_children(key)`); when passed as a
  named local variable rather than inline, it depends on a `TargetFn`
  decay fix present only in the pinned TTG fork (see
  `cmake/modules/FindOrFetchTTG.cmake`) — without it, the aggregator stores
  a dangling reference to the closure.
- A known, unresolved intermittent TTG/PaRSEC bug (segfault/hang in the
  downward pass) correlates with task/key volume and large (>8-byte) key
  types going through TTG's `parsec_key_t` pointer-indirection path. Not
  fixed — if you hit it, it's a pre-existing runtime issue, not a bug in
  this project's graph wiring.

### Multi-rank execution

The solver runs under `mpirun -n P`. All communication goes through
`exchange`/`allgather` in `src/exchange_ttg.h`: arrays of trivially
copyable elements, sent per destination rank as TTG split-metadata blocks
moved by RMA (TTG's active messages cap a serialized value at 1 MB).
`--time` reports time in these collectives as `exchange`; output (energy,
VTK) comes from rank 0, after gathering the particles there.

- **Grouped path: domain decomposition** (`src/distributed_tree.h`, see
  its header for the steps). Each rank holds only its own particles: a
  contiguous Morton key range, from weighted samples of every rank's keys
  each step. It builds the tree of its particles in the global domain
  (`build_nodes` with a `LeafRule` that keeps splitting cells reaching
  into other ranks' ranges, down to its *branches*), all ranks assemble the
  same top tree from everyone's branches, and each rank sends every other
  rank its *locally essential tree*: what that rank's walks can open,
  decided by the walk's own MAC against a hierarchy of the receiver's
  branch boxes. Groups never start above a branch, which that relies on;
  `run_grouped_pass` returns how many opened nodes lacked data (must be 0)
  and `step_particles` throws otherwise. Forces differ from a single tree
  only through cells shared between ranks (split further; targets never
  span ranks): ~1e-7 relative on the galaxy data, the Barnes-Hut error
  unchanged. `tests/test_domain_decomposition.cc` checks this (ctest runs
  it on 1 and 4 ranks). 32M bodies, 128 cores on one node, steady state:
  ~3.7 s/step on 1 rank, 3.9 s on 4x32 threads, 3.6 s on 8x16 (build
  ~0.9-1.2 s, walk ~2.6 s, exchange ~0.1 s); the first step moves almost
  every particle and is slower.
- **Per-node path: replicated data** (`src/replicated.h`): every rank holds
  all particles and builds the same tree, tasks stay on the creating rank
  (`local_keymap`), each rank starts only the target leaves in its equal
  share of the particles (`owner_of`), and `share_particles` sends the
  updated particles to all ranks after each step.

On one node, give each rank its own cores and turn off PaRSEC's binding:
```sh
PARSEC_MCA_bind_threads=0 TTG_NUM_THREADS=32 mpirun -n 4 --map-by slot:PE=32 --bind-to core \
  -x PARSEC_MCA_bind_threads -x TTG_NUM_THREADS build/src/pepc-ttg-parsec ...
```
PaRSEC prints `unknown option` for the solver's own flags; that is
harmless.

### Distributed upward pass (prototype, additive)

`src/morton.h` and `src/distributed_upward.h` implement a TTG-native (no raw
MPI) prototype of a Warren-Salmon-style distributed upward pass, kept fully
separate from the single-rank pipeline above (`upward_ttg.h`/`bh_pass.h`)
rather than modifying it:

- Each rank owns a contiguous range of the global Morton (Z-order) curve
  (`morton.h`: `morton_lo`, `range_contains`, `decompose_range`) and builds
  its own local `Tree` against one globally-shared `Domain`
  (`Tree`'s `Domain`-accepting constructor, `compute_domain()` in
  `src/domain.h`) — every rank must agree on the same `Domain` or the same
  `Key` would mean a different physical box on different ranks.
- A rank's local upward pass (`RankUpward` in `distributed_upward.h`) is
  structurally the same as `upward_ttg.h`, except each `seed_leaf`/
  `combine_up` redirects to a `branch_edge` instead of self-looping once a
  node's parent falls outside its own Morton range.
- `ExchangeGraph` bundles each rank's redirected contributions
  (`bundle_tt`, an aggregator), broadcasts the bundle to every rank, and
  each rank's `allgather_tt` folds the combined set back up to the root —
  always finalizing the *deepest* remaining key first (redirect depth can
  differ across ranks along the same ancestor chain, so processing
  level-by-level in lockstep is incorrect).
- Simulated rank r runs on process `r % P` (`process_of`):
  `tests/test_distributed_upward.cc` simulates 5 ranks in one process under
  ctest, and one rank per process under `mpirun -n P` (P > 1), where the
  exchange goes between processes (checked by hand up to P = 8). What it
  does **not** yet do: build only the local rank's tree (every process
  builds all of them) or perform the actual parallel particle-redistribution
  sort. The solver's domain decomposition (`distributed_tree.h`) does not
  use it: it exchanges branches with a plain allgather instead.

## Layout

- `src/` — library + `pepc-ttg` executable: tree build (`tree.h/.cc`;
  grouped path: `flat_tree.h`, `flat_build_ttg.h`), TTG passes
  (`upward_ttg.h`, `traversal_ttg.h`, `bh_pass.h`; grouped: `grouped_ttg.h`), multi-rank
  (`exchange_ttg.h`, `distributed_tree.h`, `replicated.h`), distributed upward
  prototype (`morton.h`, `distributed_upward.h`), physics (`multipole.h`,
  `brute_force.h`, `integrate.h`/`integrate_ttg.h`), I/O (`io.h/.cc`, CSV
  and VTK).
- `tests/` — correctness tests (Barnes-Hut vs. brute force, energy
  conservation, distributed-vs-single-tree upward pass, domain-decomposed
  vs. single-tree forces), pure-math checks
  (force kernel, Morton range math), a standalone tree-build perf probe.
- `tools/` — PEPC comparison workflow: `gen_shared_ic.py` generates matching
  initial conditions in both pepc-ttg's CSV format and PEPC's binary format;
  `compare_pepc.cc` runs pepc-ttg's own brute-force kernel on those ICs;
  `parse_pepc_vtu.py` extracts PEPC's VTU output for the same ICs so the two
  can be diffed.
- `data/` — sample initial-condition CSVs.
