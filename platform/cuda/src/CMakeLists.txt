enable_language(CUDA)

set(CudaSrc "cudaRenderer.cu"
            "tasks.cu")

add_library(cudaRenderer ${CudaSrc})