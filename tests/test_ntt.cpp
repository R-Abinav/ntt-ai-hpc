#include <iostream>
#include <vector>
#include <cassert>
#include "ntt.h"

// simple test framework
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    std::cout << "test: " << name << " ... "; \
    try

#define PASS() \
    { std::cout << "passed" << std::endl; tests_passed++; }
#define FAIL(msg) \
    { std::cout << "failed (" << msg << ")" << std::endl; tests_failed++; }

int main() {
    const uint64_t P = 7340033; // ntt-friendly prime

    // test 1: mod_pow basic
    TEST("mod_pow") {
        assert(mod_pow(2, 10, 1000) == 24);  // 1024 mod 1000
        assert(mod_pow(3, 0, 7) == 1);
        assert(mod_pow(5, 1, 13) == 5);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 2: mod_inv
    TEST("mod_inv") {
        uint64_t inv = mod_inv(3, P);
        assert((__uint128_t)3 * inv % P == 1);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 3: primitive root of unity
    TEST("primitive_root_of_unity") {
        uint64_t w = find_primitive_root_of_unity(8, P);
        assert(mod_pow(w, 8, P) == 1);
        assert(mod_pow(w, 4, P) != 1);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 4: ntt forward + inverse round-trip (length 8)
    TEST("ntt_round_trip_8") {
        std::vector<uint64_t> a = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<uint64_t> orig = a;
        ntt_forward(a, P);
        // after forward, values should differ
        assert(a != orig);
        ntt_inverse(a, P);
        assert(a == orig);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 5: ntt round-trip (length 4)
    TEST("ntt_round_trip_4") {
        std::vector<uint64_t> a = {10, 20, 30, 40};
        std::vector<uint64_t> orig = a;
        ntt_forward(a, P);
        ntt_inverse(a, P);
        assert(a == orig);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 6: polynomial multiplication
    TEST("poly_mul_ntt_basic") {
        // (1 + 2x + 3x^2) * (4 + 5x) = 4 + 13x + 22x^2 + 15x^3
        std::vector<uint64_t> a = {1, 2, 3};
        std::vector<uint64_t> b = {4, 5};
        auto c = poly_mul_ntt(a, b, P);
        std::vector<uint64_t> expected = {4, 13, 22, 15};
        assert(c == expected);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 7: polynomial multiplication (identity)
    TEST("poly_mul_ntt_identity") {
        // a(x) * 1 = a(x)
        std::vector<uint64_t> a = {7, 11, 13, 17};
        std::vector<uint64_t> b = {1};
        auto c = poly_mul_ntt(a, b, P);
        assert(c == a);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    // test 8: larger round-trip (length 16)
    TEST("ntt_round_trip_16") {
        std::vector<uint64_t> a(16);
        for (int i = 0; i < 16; i++) a[i] = (i * 37 + 5) % P;
        auto orig = a;
        ntt_forward(a, P);
        ntt_inverse(a, P);
        assert(a == orig);
        PASS();
    } catch (...) { FAIL("exception thrown"); }

    std::cout << std::endl;
    std::cout << "results: " << tests_passed << " passed, "
              << tests_failed << " failed" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
