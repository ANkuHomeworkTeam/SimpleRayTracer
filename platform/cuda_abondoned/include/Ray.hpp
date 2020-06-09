#pragma once
#ifndef __CUDA_RAY_HPP__
#define __CUDA_RAY_HPP__

#include "renderInfo.hpp"
#include "cudaOperation.hpp"
#include <cuda_runtime.h>
namespace Renderer
{
    namespace Cuda
    {
        struct Ray
        {
        public:
            Vec3 origin;
            Vec3 direction;
            __device__
            Vec3 at(float t) const {
                return add(origin, mul(direction, t));
            }
            __device__
            Ray(const Vec3& origin, const Vec3& direction):
                origin(origin), direction(direction)
            {}
        };
    }; // namespace Cuda  
}; // namespace Renderer


#endif