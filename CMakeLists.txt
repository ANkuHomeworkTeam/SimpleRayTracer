## nku 1711326 hrl 2020

cmake_minimum_required(VERSION 3.10)


project(PathTracingRenderer VERSION 1.0
                            DESCRIPTION "并行计算和计算机图形学的作业"
                            LANGUAGES CXX CUDA
                            )

enable_language(CUDA)

set(RendererBase "${PROJECT_SOURCE_DIR}/src")

configure_file(
    "${PROJECT_SOURCE_DIR}/include/config.h.in"
    "${PROJECT_BINARY_DIR}/include/config.h")

include_directories(
    "${PROJECT_SOURCE_DIR}/extern"
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
    "${RendererBase}/include")

set(CMAKE_BUILD_RPATH ".build")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")

# set(CMAKE_CUDA_FLAGS "-dc -rdc=true -c" )

add_subdirectory(platform)
add_subdirectory(src lib)
add_subdirectory(apps bin)


