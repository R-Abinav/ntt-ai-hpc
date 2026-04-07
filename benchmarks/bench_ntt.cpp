#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include "ntt.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// ntt-friendly prime: 7 * 2^20 + 1  (supports transforms up to length 2^20)
static const uint64_t P = 7340033;

// fill a polynomial with uniform random coefficients in [0, P)
static std::vector<uint64_t> make_random_poly(uint64_t size, std::mt19937_64& rng) {
    std::uniform_int_distribution<uint64_t> dist(0, P - 1);
    std::vector<uint64_t> poly(size);
    for (auto& x : poly) x = dist(rng);
    return poly;
}

// run poly_mul_ntt once and return wall-clock time in microseconds
// uses seed 42 so every call with the same half_size produces the same inputs
static double time_poly_mul(uint64_t half_size, int threads) {
    std::mt19937_64 rng(42);
    auto a = make_random_poly(half_size, rng);
    auto b = make_random_poly(half_size, rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto c   = poly_mul_ntt(a, b, P, threads);
    auto t1  = std::chrono::high_resolution_clock::now();

    // prevent the compiler from DCE-ing the result
    volatile uint64_t sink = c[0];
    (void)sink;

    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// return the median of a mutable vector (sorted in-place)
static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    std::size_t mid = v.size() / 2;
    if (v.size() % 2 == 1)
        return v[mid];
    return (v[mid - 1] + v[mid]) * 0.5;
}

int main() {
    // --- configuration -------------------------------------------------------
    // Each entry n is the internal NTT size.
    // We pass polynomials of length n/2 to poly_mul_ntt so that
    // result_size = n/2 + n/2 - 1 = n-1  →  padded to n.
    const std::vector<uint64_t> ntt_sizes = {
        64, 128, 256, 512, 1024, 2048, 4096,
        8192, 16384, 32768, 65536, 131072,
        262144, 524288, 1048576
    };
    const std::vector<int> thread_cnts = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32};
    constexpr int               RUNS        = 5;
    // -------------------------------------------------------------------------

#ifdef _OPENMP
    std::cerr << "[bench] OpenMP available — max threads = "
              << omp_get_max_threads() << "\n";
#else
    std::cerr << "[bench] OpenMP NOT available — only thread_count=1 will be run\n";
#endif

    // CSV header
    std::cout << "n,threads,time_us\n";

    for (uint64_t n : ntt_sizes) {
        const uint64_t half = n / 2;  // polynomial length fed to poly_mul_ntt

        for (int t : thread_cnts) {
#ifndef _OPENMP
            // without OpenMP every thread count > 1 is identical to 1;
            // skip to avoid duplicated rows
            if (t > 1) continue;
#endif
            // one warm-up run to prime caches / JIT effects (not recorded)
            time_poly_mul(half, t);

            // timed runs
            std::vector<double> times(RUNS);
            for (int r = 0; r < RUNS; r++)
                times[r] = time_poly_mul(half, t);

            double med = median(times);

            std::cout << n << "," << t << "," << med << "\n";
            std::cout.flush();  // stream results out line-by-line
        }
    }

    return 0;
}