#include "tasks.hpp"
#include <cuda_runtime.h>
#include <iostream>
namespace Renderer
{
namespace Cuda
{
    __global__ void renderTask(Vec3* pixels, int w, int h) {
        int pixelX = blockIdx.x;
        int pixelY = blockIdx.y;
        int sampleNum = threadIdx.x;
    }
}; // namespace Cuda
}; // namespace Renderer
