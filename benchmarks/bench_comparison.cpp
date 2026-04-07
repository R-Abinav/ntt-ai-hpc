// bench_comparison.cpp
// single-shot benchmark used by comparison.py
//
// usage: ./bench_comparison <n> <threads>
//   n       : ntt size (power of 2, 64 -- 1048576)
//   threads : thread count (1 = serial, >1 = parallel)
//
// output: one line  →  <n>,<threads>,<median_time_us>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "ntt.h"

static const uint64_t P = 7340033; // 7 * 2^20 + 1
static const int      RUNS   = 7;  // take median of 7 runs
static const int      WARMUP = 2;  // discard first 2

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: bench_comparison <n> <threads>\n";
        return 1;
    }

    const int64_t n_arg       = std::atoll(argv[1]);
    const int     thread_arg  = std::atoi(argv[2]);

    if (n_arg <= 0 || (n_arg & (n_arg - 1)) != 0) {
        std::cerr << "n must be a positive power of 2\n";
        return 1;
    }
    if (thread_arg < 1) {
        std::cerr << "threads must be >= 1\n";
        return 1;
    }

    const uint64_t n = static_cast<uint64_t>(n_arg);

    // build two random polynomials of half-size so the padded ntt is exactly n
    const uint64_t half = n / 2;
    std::vector<uint64_t> a(half), b(half);
    for (uint64_t i = 0; i < half; i++) {
        a[i] = (i * 6700417 + 31337) % P;
        b[i] = (i * 1000003 + 99991) % P;
    }

    std::vector<double> times;
    times.reserve(WARMUP + RUNS);

    for (int r = 0; r < WARMUP + RUNS; r++) {
        std::vector<uint64_t> a_copy = a;
        std::vector<uint64_t> b_copy = b;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto _c = poly_mul_ntt(a_copy, b_copy, P, thread_arg);
        auto t1 = std::chrono::high_resolution_clock::now();

        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        times.push_back(us);
    }

    // discard warmup, take median of remaining runs
    std::vector<double> valid(times.begin() + WARMUP, times.end());
    std::sort(valid.begin(), valid.end());
    double median = valid[valid.size() / 2];

    // output: n,threads,time_us  (no newline extras — easy to parse from python)
    std::cout << n << "," << thread_arg << "," << median << "\n";
    return 0;
}