#include "tasks.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include "details.hpp"
#include <stdio.h>


using namespace thrust;
namespace Renderer
{
namespace Cuda
{
    __constant__  Vec3 gpuSobolSequence[SOBOL_SEQUENCE_CYCLE];
    __constant__ RayGeneratorInfo rayGenerator;
    void initSobolSequence() {
        for(int i=0; i<SOBOL_SEQUENCE_CYCLE; i++)
            sobolSequence[i] = normalize(sobolSequence[i]);
        cudaMemcpyToSymbol(gpuSobolSequence, sobolSequence,
        sizeof(Vec3)*SOBOL_SEQUENCE_CYCLE);
    }
    __device__
    const Vec3& sobol(int index) {
        return gpuSobolSequence[index%SOBOL_SEQUENCE_CYCLE];
    }

    void initRayGenerator() {
        RayGeneratorInfo temp;
        temp.position = cam.position;
        temp.lensRadius = cam.aperture/2.f;
        
        auto theta = radians(cam.vfov);
        auto halfHeight = tan(theta/2);
        auto halfWidth = cam.aspect*halfHeight;

        Vec3 up{0.f, 1.f, 1.f};
        temp.w = normalize(sub(cam.position, cam.lookat));
        temp.u = normalize(cross(up, temp.w));
        temp.v = normalize(cross(temp.w, temp.u));

        if (cam.focusDistance<=0||cam.aperture == 0)
            cam.focusDistance = 1;
        
        temp.lowerLeft = sub(sub(
            sub(cam.position, mul(temp.u, cam.focusDistance*halfWidth)),
            mul(temp.v ,halfHeight*cam.focusDistance)),
            mul(temp.w, cam.focusDistance));

        temp.horizontal = mul(temp.u, 2.f*halfWidth*cam.focusDistance);
        temp.vertical = mul(temp.v, 2.f*halfHeight*cam.focusDistance);

        cudaMemcpyToSymbol(rayGenerator, &temp, sizeof(RayGeneratorInfo));
    }

    __device__
    Vec3 trace(const Ray& r, int depth, TextureInfo* tbs, MaterialInfo* mbs, ObjectInfo* obs, Vec3* vbs ,int size);

    __global__ void renderTask(
    Vec3* pixels,
    int w,
    int h,
    int depth,
    TextureInfo* tbs,
    MaterialInfo* mbs,
    ObjectInfo* obs,
    Vec3* vbs,
    int size) {
        int pixelX = blockIdx.x;
        int pixelY = blockIdx.y;
        int sampleNum = threadIdx.x;
        
        float u = float(float(pixelX)+sobol((sampleNum+pixelX+pixelY)%SOBOL_SEQUENCE_CYCLE).x)/float(w);
        float v = float(float(pixelY)+sobol((sampleNum+pixelX+pixelY)%SOBOL_SEQUENCE_CYCLE).y)/float(h);
        auto ray = rayGenerator.getRay(u, v, (sobol(sampleNum)));
        auto color = trace(ray, depth ,tbs, mbs, obs, vbs, size);
        // printf("Ray: ori: [ %f, %f, %f ], dir: [ %f, %f, %f ]\n", ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z);

        __syncthreads();
        auto writePixels = [&w, &h, &pixels] (int x, int y, const Vec3& rgb) -> void {
            // pixels[x + w*y].x = rgb.r;
            // pixels[x + w*y].y = rgb.g;
            // pixels[x + w*y].z = rgb.b;
            atomicAdd(&pixels[x + w*y].r, rgb.r/blockDim.x);
            atomicAdd(&pixels[x + w*y].g, rgb.g/blockDim.x);
            atomicAdd(&pixels[x + w*y].b, rgb.b/blockDim.x);
        };

        writePixels(pixelX, pixelY, color);
    }

    __device__
    HitRecord hit(const Ray& ray, const ObjectInfo& obj, Vec3* vbs, float min = 0.f, float max = FLOAT_INF ) {
        if (obj.type == ObjectType::Sphere) {
            Vec3 oc = sub(ray.origin, obj.v1);
            float a = dot(ray.direction, ray.direction);
            float b = dot(oc, ray.direction);
            float c = dot(oc, oc) - obj.f1*obj.f1;
            float discriminant = b*b - a*c;
            if (discriminant > 0) {
                float temp = (-b-std::sqrt(discriminant))/a;
                if (temp < max && temp >= min) {
                    auto hitPoint = ray.at(temp);
                    auto normal = div(sub(hitPoint, obj.v1), obj.f1);
                    if (dot(normal, ray.direction)<0) {
                        normal = neg(normal);
                    }
                    return HitRecord{{
                            .t = temp,
                            .hitPoint = hitPoint,
                            .normal = normal,
                            .material = obj.material
                        }
                    };
                }
                temp = (-b+std::sqrt(discriminant))/a;
                if (temp < max && temp >= min) {
                    auto hitPoint = ray.at(temp);
                    auto normal = div(sub(hitPoint, obj.v1), obj.f1);
                    if (dot(normal, ray.direction)<0) {
                        normal = neg(normal);
                    }
                    return HitRecord{{
                            .t = temp,
                            .hitPoint = hitPoint,
                            .normal = normal,
                            .material = obj.material
                        }
                    };
                }
            }
            return nullopt;
        }
        else if(obj.type == ObjectType::Triangle)
        {
            auto castVertex = [&obj, vbs](int i) -> const Vec3& {
                return vbs[obj.i1 + i];
            };
            auto e1 = sub(castVertex(1), castVertex(0));
            auto e2 = sub(castVertex(2), castVertex(0));
            auto normal = cross(e1, e2);
            auto P = cross(ray.direction, e2);
            float det = dot(e1, P);
            Vec3 T;
            if (det > 0) {
                T = sub(ray.origin, castVertex(0));
            }
            else {
                T = sub(castVertex(0), ray.origin);
                det = -det;
            }
            if (det < 0.00001f) {
                return nullopt;
            }
            float u, v, t;
            u = dot(T, P);
            if (u > det || u < 0.f) {
                return nullopt;
            }
            Vec3 Q = cross(T, e1);
            v = dot(ray.direction, Q);
            if (v < 0.f || v + u > det) {
                return nullopt;
            }
            t = dot(e2, Q);
            float fInvDet = 1.f / det;
            t *= fInvDet;
            if (t > max || t < min) return nullopt;
            if (dot(normal, ray.direction) > 0) {
                normal = neg(normal);
            }
            return HitRecord{{
                .t = t,
                .hitPoint = ray.at(t),
                .normal = normal,
                .material = obj.material
            }};
        }
        return nullopt;    
    }

    __device__
    HitRecord intersectionTest(const Ray& ray, ObjectInfo* obs, Vec3* vbs, int size, float min = 0.f, float max = FLOAT_INF) {
        HitRecord hitRecord = nullopt;
        float closet = max;
        for(int i = 0; i < size; i++) {
            auto record = hit(ray, obs[i], vbs, min, closet);
            if (record) {
                closet = record->t;
                hitRecord = record;
            }
        }
        return hitRecord;
    }

    __device__
    Scattered BSDF(const Vec3& hitPoint, const Vec3& normal, const Vec3& in, MaterialInfo* mbs, id_t material, TextureInfo* tbs) {
        return {
            {0.2, 0.3, 0.4},
            {{1,1,1},{1,1,1}}
        };
    }

    __device__
    Scattered shade(const Vec3& hitPoint, const Vec3& normal, const Vec3& in, MaterialInfo* mbs, id_t material, TextureInfo* tbs) {
        return BSDF(hitPoint, normal, in, mbs, material, tbs);
    }

    __device__
    Vec3 trace(const Ray& ray, int depth, TextureInfo* tbs, MaterialInfo* mbs, ObjectInfo* obs, Vec3* vbs ,int size) {
        Vec3 color{1, 1, 1};
        Ray r = ray;
        for (int i = 0; i < depth; i++) {
            auto hitRecord = intersectionTest(r, obs, vbs, size);
            if (hitRecord) {
                auto info = shade(hitRecord->hitPoint, hitRecord->normal, ray.direction, mbs, hitRecord->material, tbs);
                color = mul(color, info.attenuation);
                r = info.ray;
                printf("hit\n");
            }
            else {
                color = mul(color, {1, 1, 1});
                break;
            }
        }
        return color;
    }



    __global__ void gammaTask(Vec3* pixels, int w, int h) {
        int pixelX = blockIdx.x;
        int pixelY = blockIdx.y;
        pixels[pixelX + w*pixelY] = sqrt(pixels[pixelX + w*pixelY]);
    }
}; // namespace Cuda
}; // namespace Renderer
