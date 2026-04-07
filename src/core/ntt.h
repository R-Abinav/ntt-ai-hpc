#ifndef NTT_H
#define NTT_H

#include <cstdint>
#include <vector>

// modular arithmetic helpers
uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod);
uint64_t mod_inv(uint64_t a, uint64_t mod);

// find a primitive n-th root of unity modulo p
// p must be prime and (p - 1) must be divisible by n
uint64_t find_primitive_root_of_unity(uint64_t n, uint64_t p);

// forward ntt (in-place, cooley-tukey butterfly)
// a:            coefficient vector of length n (must be power of 2)
// p:            prime modulus where p ≡ 1 (mod n)
// thread_count: number of OpenMP threads to use (0 = system default)
void ntt_forward(std::vector<uint64_t>& a, uint64_t p, int thread_count = 0);

// inverse ntt (in-place)
// thread_count: number of OpenMP threads to use (0 = system default)
void ntt_inverse(std::vector<uint64_t>& a, uint64_t p, int thread_count = 0);

// polynomial multiplication via ntt
// returns c = a * b (mod p), result has size a.size() + b.size() - 1
// thread_count: number of OpenMP threads to use (0 = system default)
std::vector<uint64_t> poly_mul_ntt(std::vector<uint64_t> a,
                                    std::vector<uint64_t> b,
                                    uint64_t p,
                                    int thread_count = 0);

#endif // NTT_H