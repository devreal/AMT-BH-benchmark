#include "io.h"

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace bh {

namespace {

// Read-only memory map of a whole file (unmapped on destruction).
class MappedFile {
 public:
  explicit MappedFile(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open input file: " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("cannot stat input file: " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ > 0) {
      void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
      if (p == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("cannot map input file: " + path);
      }
      data_ = static_cast<const char*>(p);
      ::madvise(p, size_, MADV_SEQUENTIAL);
    }
    ::close(fd);
  }
  ~MappedFile() {
    if (data_ != nullptr) ::munmap(const_cast<char*>(data_), size_);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const char* begin() const { return data_; }
  const char* end() const { return data_ + size_; }

 private:
  const char* data_ = nullptr;
  std::size_t size_ = 0;
};

// CPUs this process may run on (respects taskset/cgroup pinning, unlike
// std::thread::hardware_concurrency).
unsigned usable_cpus() {
#if defined(__linux__)
  cpu_set_t set;
  if (sched_getaffinity(0, sizeof(set), &set) == 0) return std::max(1, CPU_COUNT(&set));
#endif
  return std::max(1u, std::thread::hardware_concurrency());
}

const char* skip_blanks(const char* p, const char* end) {
  while (p < end && (*p == ' ' || *p == '\t')) ++p;
  return p;
}

// Parses one field ending at ',' (or, for the last one, at the line end)
// and advances `p` past its separator.
template <typename T>
bool parse_field(const char*& p, const char* end, T& out, bool last) {
  p = skip_blanks(p, end);
  auto [q, ec] = std::from_chars(p, end, out);
  if (ec != std::errc()) return false;
  q = skip_blanks(q, end);
  if (last) {
    if (q < end && *q == '\r') ++q;
    if (q != end) return false;
  } else {
    if (q >= end || *q != ',') return false;
    ++q;
  }
  p = q;
  return true;
}

// Parses the lines of [begin, end) - which starts at a line start and ends
// at a line end - appending to `out`. Returns the first bad line, if any.
std::string parse_lines(const char* begin, const char* end, std::vector<Particle>& out) {
  const char* line = begin;
  while (line < end) {
    const char* eol = static_cast<const char*>(std::memchr(line, '\n', static_cast<std::size_t>(end - line)));
    if (eol == nullptr) eol = end;
    const char* p = skip_blanks(line, eol);
    if (p != eol && *p != '\r') {
      Particle q;
      if (!(parse_field(p, eol, q.id, false) && parse_field(p, eol, q.mass, false) &&
            parse_field(p, eol, q.pos.x, false) && parse_field(p, eol, q.pos.y, false) &&
            parse_field(p, eol, q.pos.z, false) && parse_field(p, eol, q.vel.x, false) &&
            parse_field(p, eol, q.vel.y, false) && parse_field(p, eol, q.vel.z, true))) {
        return std::string(line, eol);
      }
      out.push_back(q);
    }
    line = eol + 1;
  }
  return {};
}

}  // namespace

// Memory-maps the file and parses it in parallel: one chunk per usable CPU,
// with chunk boundaries moved forward to the next line start. This runs
// before ttg::execute() (see main.cc), so the threads do not compete with
// the runtime's workers. An optional header line (first field not a
// number, as in the AMT-BH data generator's output) is skipped.
std::vector<Particle> read_csv(const std::string& path) {
  MappedFile file(path);
  const char* begin = file.begin();
  const char* end = file.end();
  if (begin == end) return {};

  {
    const char* eol = static_cast<const char*>(std::memchr(begin, '\n', static_cast<std::size_t>(end - begin)));
    const char* p = skip_blanks(begin, eol ? eol : end);
    ParticleId id;
    if (std::from_chars(p, eol ? eol : end, id).ec != std::errc()) begin = eol ? eol + 1 : end;
  }

  // ~32 MiB per chunk at least, so small files stay on one thread
  std::size_t bytes = static_cast<std::size_t>(end - begin);
  unsigned n_chunks = static_cast<unsigned>(
      std::clamp<std::size_t>(bytes / (std::size_t(32) << 20), 1, usable_cpus()));
  std::vector<const char*> cut(n_chunks + 1);
  cut[0] = begin;
  cut[n_chunks] = end;
  for (unsigned c = 1; c < n_chunks; ++c) {
    const char* p = begin + bytes * c / n_chunks;
    p = std::max(p, cut[c - 1]);
    const char* eol = static_cast<const char*>(std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
    cut[c] = eol ? eol + 1 : end;
  }

  std::vector<std::vector<Particle>> parts(n_chunks);
  std::vector<std::string> errors(n_chunks);
  auto parse_chunk = [&](unsigned c) {
    // ~70 bytes per line for the generator's output; just a reservation hint
    parts[c].reserve(static_cast<std::size_t>(cut[c + 1] - cut[c]) / 64 + 16);
    errors[c] = parse_lines(cut[c], cut[c + 1], parts[c]);
  };
  {
    std::vector<std::thread> threads;
    for (unsigned c = 1; c < n_chunks; ++c) threads.emplace_back(parse_chunk, c);
    parse_chunk(0);
    for (auto& t : threads) t.join();
  }
  for (const auto& e : errors) {
    if (!e.empty()) throw std::runtime_error("malformed line in " + path + ": '" + e + "'");
  }

  if (n_chunks == 1) return std::move(parts[0]);
  std::vector<std::size_t> offset(n_chunks + 1, 0);
  for (unsigned c = 0; c < n_chunks; ++c) offset[c + 1] = offset[c] + parts[c].size();
  std::vector<Particle> particles(offset[n_chunks]);
  {
    std::vector<std::thread> threads;
    for (unsigned c = 0; c < n_chunks; ++c) {
      threads.emplace_back([&, c] { std::copy(parts[c].begin(), parts[c].end(), particles.begin() + offset[c]); });
    }
    for (auto& t : threads) t.join();
  }
  return particles;
}

void write_vtk(const std::string& path, const std::vector<Particle>& particles) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open output file: " + path);

  out << "# vtk DataFile Version 3.0\n";
  out << "pepc-ttg particle snapshot\n";
  out << "ASCII\n";
  out << "DATASET POLYDATA\n";
  out << "POINTS " << particles.size() << " double\n";
  for (const auto& p : particles) out << p.pos.x << ' ' << p.pos.y << ' ' << p.pos.z << '\n';
  out << "POINT_DATA " << particles.size() << "\n";
  out << "SCALARS mass double 1\nLOOKUP_TABLE default\n";
  for (const auto& p : particles) out << p.mass << '\n';
  out << "VECTORS velocity double\n";
  for (const auto& p : particles) out << p.vel.x << ' ' << p.vel.y << ' ' << p.vel.z << '\n';
}

}  // namespace bh
