#pragma once

#include <ttg.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <vector>

// Collective data exchange between ranks over TTG, for the multi-rank paths
// (replicated.h, distributed_tree.h): every rank sends one array of
// trivially copyable elements to each other rank. The arrays travel in
// blocks as raw bytes, moved by RMA rather than through TTG's active
// messages (see the SplitMetadataDescriptor below), which cap a serialized
// value at 1 MB.

namespace bh {

inline int this_rank() { return ttg::default_execution_context().rank(); }
inline int num_ranks() { return ttg::default_execution_context().size(); }

// One block of a rank's array for one destination: elements
// [offset, offset + count) of an array of `total` elements.
template <typename T>
struct PodBlock {
  static_assert(std::is_trivially_copyable_v<T>);
  using metadata_t = std::array<std::uint64_t, 3>;  // total, offset, count
  std::uint64_t total = 0, offset = 0, count = 0;
  std::shared_ptr<T[]> data;

  PodBlock() = default;
  // received: allocates room for the elements
  explicit PodBlock(const metadata_t& md) : total(md[0]), offset(md[1]), count(md[2]), data(new T[md[2]]) {}
  // sent: refers to the caller's elements, which must outlive the fence
  PodBlock(std::uint64_t total, std::uint64_t offset, std::uint64_t count, const T* p)
      : total(total), offset(offset), count(count), data(const_cast<T*>(p), [](T*) {}) {}

  // Full serialization for MADNESS archives - TTG walks every value with
  // one (looking for ttg::Buffer members) even when it is sent by split
  // metadata.
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & total & offset & count;
    if constexpr (Archive::is_input_archive) data.reset(new T[count]);
    ar & madness::archive::wrap(reinterpret_cast<unsigned char*>(data.get()), count * sizeof(T));
  }
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    serialize(ar);
  }
};

}  // namespace bh

namespace ttg {

template <typename T>
struct SplitMetadataDescriptor<bh::PodBlock<T>> {
  auto get_metadata(const bh::PodBlock<T>& b) { return typename bh::PodBlock<T>::metadata_t{b.total, b.offset, b.count}; }
  auto get_data(bh::PodBlock<T>& b) { return std::array<iovec, 1>({iovec{b.count * sizeof(T), b.data.get()}}); }
  auto create_from_metadata(const typename bh::PodBlock<T>::metadata_t& md) { return bh::PodBlock<T>(md); }
};

}  // namespace ttg

namespace bh {

// Collective: sends out[d] to rank d for every rank d and returns what each
// rank sent this one, by rank (out[this_rank()] is copied locally). Returns
// once every rank has received everything.
template <typename T>
std::vector<std::vector<T>> exchange(const std::vector<std::span<const T>>& out) {
  const int nranks = num_ranks(), me = this_rank();
  std::vector<std::vector<T>> in(nranks);
  in[me].assign(out[me].begin(), out[me].end());
  if (nranks == 1) return in;
  // blocks of at most 64 MB (the RMA path takes int-sized lengths)
  const std::uint64_t block_elements = std::max<std::uint64_t>(1, (std::uint64_t(64) << 20) / sizeof(T));

  // key: block number << 32 | source rank << 16 | destination rank
  using BlockKey = std::uint64_t;
  std::vector<std::mutex> resizing(nranks);
  ttg::Edge<BlockKey, PodBlock<T>> edge;
  auto receive_tt = ttg::make_tt(
      [&in, &resizing](const BlockKey& key, const PodBlock<T>& b) {
        auto src = static_cast<int>((key >> 16) & 0xffff);
        {
          std::lock_guard<std::mutex> lock(resizing[src]);
          if (in[src].size() != b.total) in[src].resize(b.total);
        }
        std::memcpy(in[src].data() + b.offset, b.data.get(), b.count * sizeof(T));
      },
      ttg::edges(edge), ttg::edges(), "receive_block");
  receive_tt->set_keymap([](const BlockKey& key) { return static_cast<int>(key & 0xffff); });
  ttg::make_graph_executable(receive_tt.get());

  auto* terminal = receive_tt->template in<0>();
  for (int dst = 0; dst < nranks; ++dst) {
    if (dst == me) continue;
    const std::uint64_t total = out[dst].size();
    for (std::uint64_t first = 0, block = 0; first < total; first += block_elements, ++block) {
      std::uint64_t n = std::min(block_elements, total - first);
      BlockKey key = block << 32 | static_cast<BlockKey>(me) << 16 | static_cast<BlockKey>(dst);
      terminal->send(key, PodBlock<T>(total, first, n, out[dst].data() + first));
    }
  }
  ttg::fence();
  return in;
}

template <typename T>
std::vector<std::vector<T>> exchange(const std::vector<std::vector<T>>& out) {
  return exchange<T>(std::vector<std::span<const T>>(out.begin(), out.end()));
}

// Collective: every rank's `mine`, by rank.
template <typename T>
std::vector<std::vector<T>> allgather(std::span<const T> mine) {
  return exchange<T>(std::vector<std::span<const T>>(num_ranks(), mine));
}

}  // namespace bh
