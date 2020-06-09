#pragma once
#ifndef __CUDA_OBJECTS_HPP__
#define __CUDA_OBJECTS_HPP__

#include "renderInfo.hpp"
#include <cuda_runtime.h>

namespace Renderer
{
    namespace Cuda
    {
        void initObjects();
        __device__
        ObjectInfo getObject(unsigned int index);
        __device__
        Vec3 getVertex(unsigned int index);
        __device__
        int getObjectNum();
    } // namespace Cuda
} // namespace Renderer


#endif