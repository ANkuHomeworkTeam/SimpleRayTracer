#pragma once
#ifndef __CUDA_DETAIL_HPP__
#define __CUDA_DETAIL_HPP__

#include "renderInfo.hpp"
#include "cudaRenderer.hpp"
#include <cuda_runtime.h>
#include "Ray.hpp"
#include "cudaOperation.hpp"

namespace Renderer
{
    namespace Cuda
    {
        __global__ 
        void renderTask(
            Vec3* pixels,
            int w, int h, int depth,
            TextureInfo* tbs,
            MaterialInfo* mbs,
            ObjectInfo* obs,
            Vec3* vbs,
            int nums
        );

        // __global__ void computeObjectAABB();

        __global__ void gammaTask(Vec3* pixels, int w, int h);
        
    }; // namespace Cuda
}; // namespace Renderer
#endif