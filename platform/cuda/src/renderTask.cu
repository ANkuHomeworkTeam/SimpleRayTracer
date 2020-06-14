#include "tasks.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include "details.hpp"
#include <thrust/optional.h>
#include <stdio.h>
#include "maths/random.hpp"
#include "maths/operation.hpp"

#include "objects.hpp"
#include "BSDFs.hpp"

using namespace thrust;

namespace Renderer
{
    namespace Cuda
    {
        __device__
        Vec3 trace(const Ray& ray);

        __global__
        void renderTask(Vec3* pixels) {
            auto castIndex = [](int x, int y) -> int {
                return x + getRenderConfig().width*(getRenderConfig().height-y-1);
            };
            //printf("[ %f, %f, %f ]\n", v.x, v.y, v.z);
            int pX = blockIdx.x;
            int pY = blockIdx.y;
            // int sampleIndex = threadIdx.x;


#       pragma region TRACE_AND_SHADE
            float u = (float(pX)+getRandom())/float(getRenderConfig().width);
            float v = (float(pY)+getRandom())/float(getRenderConfig().height);
            auto ray = getRay(u, v);
            auto color = trace(ray);
            // printf("sample -> [ %f, %f, %f ]\n", color.r, color.g, color.b);
#       pragma endregion
#       pragma region WRITE_PIXELS_COLOR
            float cx = color.x / float(getRenderConfig().sampleNums);
            float cy = color.y / float(getRenderConfig().sampleNums);
            float cz = color.z / float(getRenderConfig().sampleNums);
            atomicAdd(&pixels[castIndex(pX, pY)].x, cx);
            atomicAdd(&pixels[castIndex(pX, pY)].y, cy);
            atomicAdd(&pixels[castIndex(pX, pY)].z, cz);
            //__syncthreads();
            //color = pixels[castIndex(pX, pY)];
            //printf("final -> [ %f, %f, %f ]\n", color.r, color.g, color.b);
#       pragma endregion
        }

#   pragma region TRACE
        __device__
        Vec3 trace(const Ray& ray) {
            int depth = getRenderConfig().depth;
            auto r = ray;
            Vec3 color = { 1, 1, 1 };
            for (int i = 0; i < depth; i++) {
                if (r.direction == Vec3{ 0, 0, 0 }) {
                    break;
                }
                HitRecord hitRecord =  intersectionTest(r);
                if (hitRecord) {
                    auto scattered = shade(hitRecord->material, r, hitRecord->t, hitRecord->hitPoint, hitRecord->normal);
                    r = scattered.ray;
                    color = color * scattered.attenuation;
                }
                else {
                    color = { 0, 0, 0 };
                    break;
                }
            }
            // printf("[ %f, %f, %f ]\n", color.r, color.g, color.b);
            return color;
        }

#   pragma endregion

        __global__ void gammaTask(Vec3* pixels) {
            int w = getRenderConfig().width;
            int height = getRenderConfig().height;
            int pixelX = blockIdx.x;
            int pixelY = blockIdx.y;
            if (pixels[pixelX + w*pixelY].x > 1) pixels[pixelX + w*pixelY].x = 1;
            if (pixels[pixelX + w*pixelY].y > 1) pixels[pixelX + w*pixelY].y = 1;
            if (pixels[pixelX + w*pixelY].z > 1) pixels[pixelX + w*pixelY].z = 1;
            pixels[pixelX + w*pixelY] = sqrt(pixels[pixelX + w*pixelY]);
        }
    } // namespace Cuda
} // namespace Renderer
 