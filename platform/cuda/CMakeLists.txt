cmake_minimum_required(VERSION 3.11)
enable_language(CUDA)

# set(CudaSrc "cudaRenderer.cu"
#             "tasks.cu")

aux_source_directory(./src CudaSrc)
aux_source_directory(./src/maths CudaSrc)

add_library(cudaRenderer STATIC ${CudaSrc})

set_target_properties( cudaRenderer
PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cudaRenderer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)