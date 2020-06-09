#pragma once
#ifndef __CUDA_TASKS_HPP__
#define __CUDA_TASKS_HPP__

#include "renderer.hpp"
#include <cuda_runtime.h>

namespace Renderer
{
    namespace Cuda
    {
        __global__
        void renderTask(Vec3* pixels);
    } // namespace Cuda
    
} // namespace Renderer


#endif