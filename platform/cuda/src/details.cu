#include "details.hpp"
#include "maths/random.hpp"
#include <stdio.h>
namespace Renderer
{
    namespace Cuda
    {
#   pragma region RENDER_CONFIG
        __constant__
        RenderConfig gpuRenderConfig;

        void initRenderConfig() {
            cudaMemcpyToSymbol(gpuRenderConfig, &renderConfig, sizeof(renderConfig));
        }

        __device__
        const RenderConfig& getRenderConfig() {
            return gpuRenderConfig;
        }
#   pragma endregion

#   pragma region RAY_GENERATOR
        __constant__
        RayGeneratorInfo rayGenerator;

        void initRayGenerator() {
            RayGeneratorInfo rg;
            rg.position = cam.position;
            rg.lenRadius = cam.aperture / 2.f;
            auto vfov = cam.vfov;
            if (vfov > 160) vfov = 160;
            else if (vfov < 20) vfov = 20;

            auto theta = radians(vfov);
            auto halfHeight = tan(theta / 2.f);
            auto halfWidth = cam.aspect*halfHeight;

            Vec3 up{0.f, 1.f, 0.f};
            rg.w = normalize(cam.position - cam.lookat);
            rg.u = normalize(cross(up, rg.w));
            rg.v = cross(rg.w,  rg.u);

            if (cam.focusDistance <= 0 || cam.aperture == 0) {
                cam.focusDistance = 1;
            }

            rg.lowerLeft = cam.position - halfWidth*cam.focusDistance*rg.u
                         - halfHeight*cam.focusDistance*rg.v
                         - cam.focusDistance*rg.w;
            rg.horizontal = 2*halfWidth*cam.focusDistance*rg.u;
            rg.vertical = 2*halfHeight*cam.focusDistance*rg.v;

            cudaMemcpyToSymbol(rayGenerator, &rg, sizeof(RayGeneratorInfo));
        }
        __device__
        Ray getRay(float s, float t) {
            float rdx = getSobol(threadIdx.x).x*rayGenerator.lenRadius;
            float rdy = getSobol(threadIdx.x).y*rayGenerator.lenRadius;
            Vec3 offset = rayGenerator.u*rdx + rayGenerator.v*rdy;
            return Ray{
                rayGenerator.position + offset,
                normalize(rayGenerator.lowerLeft
                + s*rayGenerator.horizontal
                + t*rayGenerator.vertical
                - rayGenerator.position -  offset)
            };
        }
#   pragma endregion
    }
}