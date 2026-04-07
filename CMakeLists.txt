cmake_minimum_required(VERSION 3.16)
project(ntt_ai_hpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# on macOS, Apple Clang does not bundle OpenMP — point CMake at Homebrew's libomp
# before calling find_package so the finder picks up the right flags and library.
if(APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    include_directories("/opt/homebrew/opt/libomp/include")
endif()

# find OpenMP (optional — serial fallback is preserved via #ifdef _OPENMP guards)
find_package(OpenMP)

# core ntt library
add_library(ntt_core
    src/core/ntt.cpp
)
target_include_directories(ntt_core PUBLIC src/core)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ntt_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — parallel NTT enabled (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "OpenMP not found — building serial fallback")
endif()

# main demo executable
add_executable(ntt_main src/main.cpp)
target_link_libraries(ntt_main PRIVATE ntt_core)

# correctness test executable
add_executable(ntt_test tests/test_ntt.cpp)
target_link_libraries(ntt_test PRIVATE ntt_core)

# benchmark executable
add_executable(bench_ntt benchmarks/bench_ntt.cpp)
target_link_libraries(bench_ntt PRIVATE ntt_core)
