#include <iostream>
#include <vector>
#include <chrono>
#include "ntt.h"

int main() {
    // ntt-friendly prime: 7340033 = 7 * 2^20 + 1
    // supports ntt up to length 2^20
    const uint64_t P = 7340033;

    std::cout << "=== ntt computation (hybrid ai-hpc project) ===" << std::endl;
    std::cout << "prime modulus p = " << P << std::endl;
    std::cout << std::endl;

    // --- demo 1: forward and inverse ntt ---
    std::cout << "--- demo 1: forward + inverse ntt ---" << std::endl;

    std::vector<uint64_t> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t n = coeffs.size();

    std::cout << "input coefficients: ";
    for (auto x : coeffs) std::cout << x << " ";
    std::cout << std::endl;

    // forward ntt
    std::vector<uint64_t> ntt_result = coeffs;
    auto t1 = std::chrono::high_resolution_clock::now();
    ntt_forward(ntt_result, P);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "ntt forward result: ";
    for (auto x : ntt_result) std::cout << x << " ";
    std::cout << std::endl;

    double fwd_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    std::cout << "forward ntt time: " << fwd_us << " us" << std::endl;

    // inverse ntt (should recover original coefficients)
    std::vector<uint64_t> recovered = ntt_result;
    auto t3 = std::chrono::high_resolution_clock::now();
    ntt_inverse(recovered, P);
    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "inverse ntt result: ";
    for (auto x : recovered) std::cout << x << " ";
    std::cout << std::endl;

    double inv_us = std::chrono::duration<double, std::micro>(t4 - t3).count();
    std::cout << "inverse ntt time: " << inv_us << " us" << std::endl;

    // verify round-trip
    bool match = (coeffs == recovered);
    std::cout << "round-trip check: " << (match ? "passed" : "failed") << std::endl;
    std::cout << std::endl;

    // --- demo 2: polynomial multiplication via ntt ---
    std::cout << "--- demo 2: polynomial multiplication via ntt ---" << std::endl;

    // a(x) = 1 + 2x + 3x^2
    std::vector<uint64_t> poly_a = {1, 2, 3};
    // b(x) = 4 + 5x
    std::vector<uint64_t> poly_b = {4, 5};

    std::cout << "a(x) = ";
    for (size_t i = 0; i < poly_a.size(); i++) {
        if (i > 0) std::cout << " + ";
        std::cout << poly_a[i];
        if (i > 0) std::cout << "x^" << i;
    }
    std::cout << std::endl;

    std::cout << "b(x) = ";
    for (size_t i = 0; i < poly_b.size(); i++) {
        if (i > 0) std::cout << " + ";
        std::cout << poly_b[i];
        if (i > 0) std::cout << "x^" << i;
    }
    std::cout << std::endl;

    auto t5 = std::chrono::high_resolution_clock::now();
    auto product = poly_mul_ntt(poly_a, poly_b, P);
    auto t6 = std::chrono::high_resolution_clock::now();

    // expected: (1+2x+3x^2)(4+5x) = 4 + 13x + 22x^2 + 15x^3
    std::cout << "a(x) * b(x) = ";
    for (size_t i = 0; i < product.size(); i++) {
        if (i > 0) std::cout << " + ";
        std::cout << product[i];
        if (i > 0) std::cout << "x^" << i;
    }
    std::cout << std::endl;

    double mul_us = std::chrono::duration<double, std::micro>(t6 - t5).count();
    std::cout << "poly multiply time: " << mul_us << " us" << std::endl;

    std::vector<uint64_t> expected = {4, 13, 22, 15};
    bool mul_ok = (product == expected);
    std::cout << "multiplication check: " << (mul_ok ? "passed" : "failed") << std::endl;

    return (match && mul_ok) ? 0 : 1;
}
