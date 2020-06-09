#pragma once
#ifndef __CUDA_BSDFS_HPP__
#define __CUDA_BSDFS_HPP__

#include "renderInfo.hpp"
#include <cuda_runtime.h>

namespace Renderer
{
    namespace Cuda
    {
        void initMaterial();
        void initTexture();
        __device__
        MaterialInfo getMaterial(unsigned int index);
        __device__
        TextureInfo getTexture(unsigned int index);
    } // namespace Cuda
} // namespace Renderer


#endif