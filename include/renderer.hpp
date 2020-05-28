#pragma once

#ifndef __RENDERER_H__
#define __RENDERER_H__

#include "renderError.hpp"

namespace Renderer
{
    enum class RenderEnv
    {
        UNDEFINE,
        CPU,
        CPU_OPTIMIZED,
        CUDA
    };

    ErrorCode init(RenderEnv env);
    ErrorCode render();

    using id_t = unsigned int;

    struct alignas(16) Vec3
    {
        union
        {
            struct
            {
                float x, y, z;
            };
            struct
            {
                float r, g, b;
            };
        };
        
    };
    

    namespace Material
    {
        id_t createLambertain(id_t texture);
    };

    namespace Object
    {
        id_t createSphere(const Vec3& position, float radius, id_t material);
        id_t createTriangle(const Vec3& p1, const Vec3& p2, const Vec3& p3);
    };

    namespace Texture{
        id_t createSolid(const Vec3& rgb);
    };

};

#endif