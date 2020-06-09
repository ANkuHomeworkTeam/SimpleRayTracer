#pragma once
#ifndef __CUDA_RENDERER_HPP__
#define __CUDA_RENDERER_HPP__

#include "renderInfo.hpp"

namespace Renderer
{
    namespace Cuda
    {
        bool checkCudaSupport();
        ErrorCode cudaRender(Vec3** pixels);
    };
};

#endif