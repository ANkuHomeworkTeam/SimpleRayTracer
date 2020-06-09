aux_source_directory(./src BaseSrc)

add_library(rendererBase ${BaseSrc})

set_property(TARGET rendererBase 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(rendererBase cudaRenderer)
target_include_directories(rendererBase PRIVATE "${PROJECT_SOURCE_DIR}/platform")
