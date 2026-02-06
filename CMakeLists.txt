cmake_minimum_required(VERSION 3.18)
project(CUDA_Project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-O3 ${CMAKE_CUDA_FLAGS}")

include(cmake/cuda.cmake)

set(CMAKE_BUILD_TYPE Release)

add_subdirectory(hello_world)
add_subdirectory(cuda_memory)