#pragma once

#ifndef __CUDA_DETAILS_HPP__
#define __CUDA_DETAILS_HPP__

#include "renderInfo.hpp"
#include <cuda_runtime.h>
#include "sobol.hpp"
#include "cudaOperation.hpp"
#include <thrust/optional.h>
#include <thrust/limits.h>

#define FLOAT_INF (thrust::numeric_limits<float>::infinity())

namespace Renderer
{
    namespace Cuda
    {
        void initSobolSequence();
        void initRayGenerator();

        Vec3* initVertex();

        TextureInfo* initTexture();

        MaterialInfo* initMaterial();

        ObjectInfo* initObject ();

        struct RayGeneratorInfo
        {
            float   lensRadius;
            Vec3    u, v, w;
            Vec3    vertical;
            Vec3    horizontal;
            Vec3    lowerLeft;
            Vec3    position;
            __device__
            Ray getRay(float s, float t, const Vec3& random) const {
                Vec3 rd = mul(random, lensRadius);
                Vec3 offset = add(mul(u,rd.x), mul(v, rd.y));
                return Ray(
                    add(position, offset),
                    add(lowerLeft, sub(sub(add(mul(horizontal, s), mul(vertical, t)), position), offset))
                );
            }
        };
        
        struct HitRecordBase {
            float   t        = FLOAT_INF;
            Vec3    hitPoint = Vec3{0, 0, 0};
            Vec3    normal   = Vec3{0, 0, 0};
            id_t    material = 0;
        };

        using HitRecord = thrust::optional<HitRecordBase>;

        struct Scattered
        {
            Vec3 attenuation;
            Ray  ray;
        };
        
    }; // namespace Cuda
    
}; // namespace Renderer


#endif