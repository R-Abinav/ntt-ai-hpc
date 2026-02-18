cmake_minimum_required(VERSION 3.16)
project(ntt_ai_hpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# core ntt library
add_library(ntt_core
    src/core/ntt.cpp
)
target_include_directories(ntt_core PUBLIC src/core)

# main executable
add_executable(ntt_main src/main.cpp)
target_link_libraries(ntt_main PRIVATE ntt_core)

# test executable
add_executable(ntt_test tests/test_ntt.cpp)
target_link_libraries(ntt_test PRIVATE ntt_core)
