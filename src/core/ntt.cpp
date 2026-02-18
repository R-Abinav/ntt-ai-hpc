#include "ntt.h"
#include <stdexcept>
#include <algorithm>

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

    // find a generator of the multiplicative group
    // try small candidates
    for (uint64_t g = 2; g < p; g++) {
        // candidate root of unity: g^((p-1)/n) mod p
        uint64_t w = mod_pow(g, (p - 1) / n, p);
        // check it is a primitive n-th root (not a root of any smaller order dividing n)
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
void ntt_forward(std::vector<uint64_t>& a, uint64_t p) {
    uint64_t n = a.size();
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("array length must be a power of 2");

    bit_reverse_permute(a);

    for (uint64_t len = 2; len <= n; len <<= 1) {
        uint64_t w = find_primitive_root_of_unity(len, p);
        for (uint64_t i = 0; i < n; i += len) {
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

// inverse ntt
void ntt_inverse(std::vector<uint64_t>& a, uint64_t p) {
    uint64_t n = a.size();

    // perform forward ntt
    ntt_forward(a, p);

    // reverse elements [1..n-1] to get inverse
    std::reverse(a.begin() + 1, a.end());

    // scale by 1/n mod p
    uint64_t n_inv = mod_inv(n, p);
    for (auto& x : a)
        x = (__uint128_t)x * n_inv % p;
}

// polynomial multiplication: c = a * b mod p using ntt
std::vector<uint64_t> poly_mul_ntt(std::vector<uint64_t> a,
                                    std::vector<uint64_t> b,
                                    uint64_t p) {
    uint64_t result_size = a.size() + b.size() - 1;

    // pad to next power of 2
    uint64_t n = 1;
    while (n < result_size) n <<= 1;
    a.resize(n, 0);
    b.resize(n, 0);

    // forward ntt on both
    ntt_forward(a, p);
    ntt_forward(b, p);

    // pointwise multiplication
    std::vector<uint64_t> c(n);
    for (uint64_t i = 0; i < n; i++)
        c[i] = (__uint128_t)a[i] * b[i] % p;

    // inverse ntt to get result
    ntt_inverse(c, p);

    c.resize(result_size);
    return c;
}
