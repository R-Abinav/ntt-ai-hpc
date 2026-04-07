#!/bin/bash
set -e
BUILD_DIR="build"

echo "configuring cmake..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "building..."
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo "build complete. executables in $BUILD_DIR/"
echo ""
echo "run main:  ./$BUILD_DIR/ntt_main"
echo "run tests: ./$BUILD_DIR/ntt_test"
echo "run bench: ./$BUILD_DIR/bench_ntt"
echo "run demo: ./$BUILD_DIR/stark_demo"
echo ""
echo "capture benchmark results as csv:"
echo "  ./$BUILD_DIR/bench_ntt > results.csv"
