#include "BSDFs.hpp"
#include "maths/random.hpp"
#include <stdio.h>
#include "objects.hpp"

namespace Renderer
{
    namespace Cuda
    {
        __device__ MaterialInfo* gpuMaterialBuffer;
        __device__ TextureInfo* gpuTextureBuffer;

        void initMaterial() {
            MaterialInfo* tmp;
            cudaMalloc((void**)&tmp, sizeof(MaterialInfo)*materialBuffer.size());
            cudaMemcpy(tmp, &materialBuffer[0], sizeof(MaterialInfo)*materialBuffer.size(), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(gpuMaterialBuffer, &tmp, sizeof(MaterialInfo*));    
        }
        void initTexture() {
            TextureInfo* tmp;
            cudaMalloc((void**)&tmp, sizeof(TextureInfo)*textureBuffer.size());
            cudaMemcpy(tmp, &textureBuffer[0], sizeof(TextureInfo)* textureBuffer.size(), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(gpuTextureBuffer, &tmp, sizeof(TextureInfo*));
        }

        __device__
        const MaterialInfo& getMaterial(unsigned int index) {
            return gpuMaterialBuffer[index];
        }
        __device__
        const TextureInfo& getTexture(unsigned int index) {
            return gpuTextureBuffer[index];
        }

#   pragma region SHADE_AND_BSDF
        __device__
        Vec3 sampleTexture(id_t texture, float u, float v) {
            const TextureInfo& tx = getTexture(texture);
            switch (tx.type)
            {
            case TextureType::Solid:
                return tx.v1;
            default:
                return {1, 1, 1};
            }
        }


        __device__ inline
        Vec3 emitted(id_t material, float t, float maxDistance) {
            const MaterialInfo& m = getMaterial(material);
            float d = maxDistance - t;
            d = d > 0.f ? d : 0.f;
            return sampleTexture(m.texture, 0, 0) * m.luminance * std::pow(d, m.luminanceAttenuation);
        }

        __device__
        Vec3 BRDF(id_t material, MaterialType type, const Vec3& hitPoint, const Vec3& in, const Vec3& out) {
            const MaterialInfo& m = getMaterial(material);
            switch (type)
            {
            case MaterialType::LAMBERTAIN:
            case MaterialType::SPECULAR:
                return sampleTexture(m.texture, 0, 0) / M_PI;
            default:
                break;
            }
            
            return { 0, 0, 0 };
        }

        __device__
        Scattered shade(id_t material, const Ray& ray, float t, const Vec3& hitPoint, const Vec3& normal) {
            const MaterialInfo& m = getMaterial(material);
            Vec3 out = { 0, 0, 0 };
            Vec3 in = -ray.direction;
            Vec3 sample = {0, 0, 0};
            bool isDirect = false;
            id_t directObj = 0;
            switch (m.type)
            {
            case MaterialType::LAMBERTAIN:
                if (getRandom() > 0.45f) {
                    out = normalize(normal + getRandomNormalizedVec3());
                    sample = BRDF(material, MaterialType::LAMBERTAIN, hitPoint, in, out) * M_PI *  2 * cos(out, normal);
                }
                else {
                    isDirect = true;
                    auto lightSampling = sampleRandomLight(hitPoint, normal);
                    if (lightSampling) {   
                        out = normalize(lightSampling->samplePoint - hitPoint);
                        directObj = lightSampling->objId;
                        sample = BRDF(material, MaterialType::LAMBERTAIN, hitPoint, in, out) * M_PI;
                    }
                }
                break;
            case MaterialType::PHONG:
                if (getRandom() > 0.5) {

                }
                else {
                    
                }
                break;
            case MaterialType::EMITTED:
                break;
            case MaterialType::SPECULAR:
                out = normalize((ray.direction - 2*dot(ray.direction, normal)*normal))+getSobolNormalized(threadIdx.x)*m.glossy;
                sample = BRDF(material, MaterialType::SPECULAR, hitPoint, in, out) * M_PI ;// (cos(out, normal)*0.9 + 0.1);
                break;
            default:
                break;
            }
            Vec3 attenuation = emitted(material, t, m.luminanceDistance) + sample;
            // printf("sample: [ %f, %f, %f ]\n", attenuation.x, attenuation.y, attenuation.z);
            return {
                attenuation,
                Ray{hitPoint, out},
                isDirect,
                directObj
            };
        }

#   pragma endregion
    } // namespace Cuda
} // namespace Renderer
