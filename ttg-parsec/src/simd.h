#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX512F__) || defined(__AVX__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace bh::simd {

// A minimal double vector of the native register width for the force kernel
// (grouped_ttg.h), built on GCC/Clang vector extensions so the arithmetic is
// portable and only the reciprocal-square-root estimate needs per-ISA code.
// It must not be wider than a hardware register: GCC lowers wider vector
// extension types generically, which on NEON kept the kernel's
// accumulators on the stack and scalarized its comparisons (~3x slower).
// Callers unroll for instruction-level parallelism instead.
#if defined(__AVX512F__)
constexpr int kWidth = 8;
#elif defined(__AVX__)
constexpr int kWidth = 4;
#else
constexpr int kWidth = 2;  // NEON, SSE2
#endif
// vector registers, for choosing how far callers unroll
#if defined(__AVX512F__) || defined(__aarch64__)
constexpr int kRegisters = 32;
#else
constexpr int kRegisters = 16;
#endif

using vdouble = double __attribute__((vector_size(kWidth * sizeof(double))));
using vmask = std::int64_t __attribute__((vector_size(kWidth * sizeof(double))));

inline vdouble broadcast(double x) { return vdouble{} + x; }

inline vdouble load(const double* p) {
  vdouble v;
  std::memcpy(&v, p, sizeof(v));  // unaligned load
  return v;
}

inline double hsum(vdouble v) {
  double s = 0.0;
  for (int i = 0; i < kWidth; ++i) s += v[i];
  return s;
}

// v where mask is set, 0 elsewhere
inline vdouble select(vmask mask, vdouble v) { return reinterpret_cast<vdouble>(mask & reinterpret_cast<vmask>(v)); }

// Newton-Raphson step for 1/sqrt(x), roughly doubling the correct bits of y.
inline vdouble rsqrt_step(vdouble x, vdouble y) { return y * (1.5 - (0.5 * x) * y * y); }

// 1/sqrt(x) for x > 0: a hardware estimate refined by Newton-Raphson
// instead of a sqrt and a divide, which are slow and poorly pipelined on
// most cores. The number of steps per ISA is set by the estimate's
// precision: AVX-512 (14 bits) and AVX (12 bits, via float) reach full
// double precision after two steps. NEON's 8-bit estimate would need a
// third step for that; two leave a relative error of <~2e-9 in 1/r^3
// (~5e-11 in test_bh_vs_bruteforce), orders of magnitude below the
// Barnes-Hut error at any useful theta, and save ~10% of the force kernel.
inline vdouble rsqrt(vdouble x) {
#if defined(__AVX512F__)
  vdouble y = reinterpret_cast<vdouble>(_mm512_rsqrt14_pd(reinterpret_cast<__m512d>(x)));
  y = rsqrt_step(x, y);
  return rsqrt_step(x, y);
#elif defined(__AVX__)
  vdouble y = reinterpret_cast<vdouble>(
      _mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(reinterpret_cast<__m256d>(x)))));
  y = rsqrt_step(x, y);
  return rsqrt_step(x, y);
#elif defined(__aarch64__) && defined(__ARM_NEON)
  float64x2_t xn = reinterpret_cast<float64x2_t>(x);
  float64x2_t y = vrsqrteq_f64(xn);
  y = vmulq_f64(y, vrsqrtsq_f64(vmulq_f64(xn, y), y));  // vrsqrtsq_f64(a, b) = (3 - a b) / 2
  y = vmulq_f64(y, vrsqrtsq_f64(vmulq_f64(xn, y), y));
  return reinterpret_cast<vdouble>(y);
#else
  vdouble y;
  for (int i = 0; i < kWidth; ++i) y[i] = 1.0 / std::sqrt(x[i]);
  return y;
#endif
}

}  // namespace bh::simd
