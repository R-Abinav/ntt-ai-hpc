# NTT-AI-HPC

Hybrid AI-HPC approach to accelerating Number Theoretic Transforms (NTT) for post-quantum cryptography.

**Course:** High Performance Computing (CS5013)  
**Author:** R. Abinav (ME23B1004)

## Overview

NTTs are core operations in lattice-based cryptographic schemes like [CRYSTALS-Kyber](https://github.com/pq-crystals/kyber) and [CRYSTALS-Dilithium](https://github.com/pq-crystals/dilithium). This project combines HPC techniques (parallelism, GPU acceleration) with AI/ML-driven optimisation to accelerate NTT computations.

## Project Structure

```
ntt-ai-hpc/
├── CMakeLists.txt
├── src/
│   ├── core/
│   │   ├── ntt.h              # ntt declarations
│   │   └── ntt.cpp            # ntt implementation
│   └── main.cpp               # demo with timing
├── tests/
│   └── test_ntt.cpp           # correctness tests
├── benchmarks/
│   └── bench_ntt.cpp          # benchmarks (wip)
├── python/
│   ├── requirements.txt
│   └── api/                   # ai api layer (wip)
└── scripts/
    └── build.sh               # build helper
```

## What's Implemented

- **Forward NTT** — iterative Cooley-Tukey butterfly with bit-reverse permutation
- **Inverse NTT** — forward NTT + reversal + scaling by 1/n
- **Polynomial multiplication** via NTT
- **Modular arithmetic** — fast exponentiation, inverse via Fermat's little theorem, primitive root of unity finder
- NTT-friendly prime: `P = 7340033` (= 7 × 2²⁰ + 1), supports transforms up to length 2²⁰

## Build & Run

```bash
# build
bash scripts/build.sh

# run demo
./build/ntt_main

# run tests (8 tests)
./build/ntt_test
```

Requires CMake ≥ 3.16 and a C++17 compiler.

## References

- [CRYSTALS-Dilithium](https://github.com/pq-crystals/dilithium)
- [CRYSTALS-Kyber](https://github.com/pq-crystals/kyber)
- [Icicle — GPU-accelerated crypto primitives](https://github.com/ingonyama-zk/icicle)
