#include "ntt.h"
#include <stdexcept>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// fast modular exponentiation: base^exp mod m
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1)
            result = (__uint128_t)result * base % mod;
        exp >>= 1;
        base = (__uint128_t)base * base % mod;
    }
    return result;
}

// modular inverse via fermat's little theorem (p must be prime)
uint64_t mod_inv(uint64_t a, uint64_t mod) {
    return mod_pow(a, mod - 2, mod);
}

// find a primitive n-th root of unity modulo p
// we need g such that g^n ≡ 1 (mod p) and g^(n/2) ≢ 1 (mod p)
uint64_t find_primitive_root_of_unity(uint64_t n, uint64_t p) {
    if ((p - 1) % n != 0)
        throw std::invalid_argument("p - 1 must be divisible by n");

    for (uint64_t g = 2; g < p; g++) {
        uint64_t w = mod_pow(g, (p - 1) / n, p);
        if (mod_pow(w, n, p) == 1 && mod_pow(w, n / 2, p) != 1) {
            return w;
        }
    }
    throw std::runtime_error("no primitive root of unity found");
}

// bit-reverse permutation of array indices
static void bit_reverse_permute(std::vector<uint64_t>& a) {
    uint64_t n = a.size();
    for (uint64_t i = 1, j = 0; i < n; i++) {
        uint64_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }
}

// forward ntt using iterative cooley-tukey butterfly
// thread_count: number of OpenMP threads to use (0 = use OpenMP default / serial if unavailable)
void ntt_forward(std::vector<uint64_t>& a, uint64_t p, int thread_count) {
    uint64_t n = a.size();
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("array length must be a power of 2");

    bit_reverse_permute(a);

#ifdef _OPENMP
    int nthreads = (thread_count > 0) ? thread_count : omp_get_max_threads();
#endif

    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t w = find_primitive_root_of_unity(len, p);
        int64_t  num_groups = static_cast<int64_t>(n / len);

        // each group [i .. i+len) is fully independent — safe to parallelise
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
        for (int64_t g = 0; g < num_groups; g++) {
            uint64_t i  = static_cast<uint64_t>(g) * len;
            uint64_t wn = 1;
            for (uint64_t j = 0; j < len / 2; j++) {
                uint64_t u = a[i + j];
                uint64_t v = (__uint128_t)a[i + j + len / 2] * wn % p;
                a[i + j]           = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
                wn = (__uint128_t)wn * w % p;
            }
        }
    }
}

// inverse ntt (in-place)
// thread_count: forwarded to internal ntt_forward call and scaling loop
void ntt_inverse(std::vector<uint64_t>& a, uint64_t p, int thread_count) {
    uint64_t n = a.size();

    // perform forward ntt
    ntt_forward(a, p, thread_count);

    // reverse elements [1..n-1] to obtain the inverse permutation
    std::reverse(a.begin() + 1, a.end());

    // scale by 1/n mod p
    uint64_t n_inv = mod_inv(n, p);

#ifdef _OPENMP
    int nthreads = (thread_count > 0) ? thread_count : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++)
        a[i] = (__uint128_t)a[i] * n_inv % p;
}

// polynomial multiplication: c = a * b mod p using ntt
// thread_count: forwarded to all internal ntt calls and the pointwise multiply
std::vector<uint64_t> poly_mul_ntt(std::vector<uint64_t> a,
                                    std::vector<uint64_t> b,
                                    uint64_t p,
                                    int thread_count) {
    uint64_t result_size = a.size() + b.size() - 1;

    // pad both inputs to the next power of 2
    uint64_t n = 1;
    while (n < result_size) n <<= 1;
    a.resize(n, 0);
    b.resize(n, 0);

    // forward ntt on both operands
    ntt_forward(a, p, thread_count);
    ntt_forward(b, p, thread_count);

    // pointwise multiplication in the frequency domain
    std::vector<uint64_t> c(n);

#ifdef _OPENMP
    int nthreads = (thread_count > 0) ? thread_count : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(nthreads)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++)
        c[i] = (__uint128_t)a[i] * b[i] % p;

    // inverse ntt to recover the product coefficients
    ntt_inverse(c, p, thread_count);

    c.resize(result_size);
    return c;
}