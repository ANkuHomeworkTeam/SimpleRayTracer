#pragma once
#ifndef __CUDA_RANDOM_HPP__
#define __CUDA_RANDOM_HPP__

#include <cuda_runtime.h>
#include "renderer.hpp"

namespace Renderer
{
    namespace Cuda
    {
        void initRandom(int threadNum);
        __device__
        float getRandom();
        __device__
        Vec3 getSobol(int index);
        __device__
        Vec3 getSobolNormalized(int index);
    } // namespace Cuda
} // namespace Renderer


#endif