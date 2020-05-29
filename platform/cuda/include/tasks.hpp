#pragma once
#ifndef __CUDA_DETAIL_HPP__
#define __CUDA_DETAIL_HPP__

#include "renderInfo.hpp"
#include "cudaRenderer.hpp"
#include <cuda_runtime.h>

namespace Renderer
{
    namespace Cuda
    {
        __global__ 
        void renderTask(
            Vec3* pixels,
            int w, int h
        );
        
    }; // namespace Cuda
}; // namespace Renderer
#endif