## nku 1711326 hrl 2020

cmake_minimum_required(VERSION 3.8)


project(CudaBasedPathTracingRenderer VERSION 1.0
                                     DESCRIPTION "并行计算和计算机图形学的作业"
                                     )

configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/include/config.h")

include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
    "${PROJECT_BINARY_DIR}/src/include")

set(CMAKE_BUILD_RPATH ".build")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_subdirectory(src lib)
add_subdirectory(apps bin)