#pragma once
#ifndef __DETAIL_HPP__
#define __DETAIL_HPP__

#include "renderInfo.hpp"
#include "cudaOperation.hpp"
#include <cuda_runtime.h>
namespace Renderer
{
    namespace Cuda
    {
        struct Ray
        {
            Vec3 origin;
            Vec3 direction;
            __device__
            Vec3 at() {
                
            }
        };
    }; // namespace Cuda  
}; // namespace Renderer


#endif