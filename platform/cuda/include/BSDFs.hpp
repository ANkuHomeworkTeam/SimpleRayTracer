#pragma once
#ifndef __CUDA_BSDFS_HPP__
#define __CUDA_BSDFS_HPP__

#include "renderInfo.hpp"
#include "details.hpp"
#include <cuda_runtime.h>

namespace Renderer
{
    namespace Cuda
    {
        void initMaterial();
        void initTexture();
        __device__
        const MaterialInfo& getMaterial(unsigned int index);
        __device__
        const TextureInfo& getTexture(unsigned int index);

        struct Scattered
        {
            Vec3 attenuation;
            Ray ray;
        };

        __device__
        Scattered shade(id_t material, const Ray& ray, float t, const Vec3& hitPoint, const Vec3& normal);
    } // namespace Cuda
} // namespace Renderer


#endif