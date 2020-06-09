#pragma once
#ifndef __CUDA_DETAILS_HPP__
#define __CUDA_DETAILS_HPP__

#include "Ray.hpp"
#include "renderInfo.hpp"

namespace Renderer
{
    namespace Cuda
    {
        struct RayGeneratorInfo
        {
            float lenRadius;
            Vec3 u, v, w;
            Vec3 vertical;
            Vec3 horizontal;
            Vec3 lowerLeft;
            Vec3 position;
        };

        void initRayGenerator();
        void initRenderConfig();
        __device__
        const RenderConfig& getRenderConfig();

    } // namespace Cuda
} // namespace Renderer


#endif