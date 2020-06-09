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

    struct RenderConfig {
        int width = 960;
        int height = 540;
        bool gamma = true;
        int depth = 3;
        int sampleNums = 256;
    };

    ErrorCode init(RenderEnv env);

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
            float d[3];
        };
        
    };
    
    ErrorCode render(Vec3** pixels);

    namespace Material
    {
        id_t createLambertain(id_t texture);
    };

    namespace Object
    {
        id_t createSphere(const Vec3& position, float radius, id_t material);
        id_t createTriangle(const Vec3& p1, const Vec3& p2, const Vec3& p3, id_t material);
    };

    namespace Texture{
        id_t createSolid(const Vec3& rgb);
    };

    struct Camera
    {
        float vfov;
        float aspect;
        float focusDistance;
        float aperture;
        Vec3 position;
        Vec3 lookat;
    };

    void setCamera(float vfov, float aspect, float focusDistance,
        float aperture, Vec3 position, Vec3 lookat);

    void setRenderConfig(int width, int height, bool gamma = true);

};

#endif