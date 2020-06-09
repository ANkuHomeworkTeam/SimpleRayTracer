#include "BSDFs.hpp"

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
        MaterialInfo getMaterial(unsigned int index) {
            return gpuMaterialBuffer[index];
        }
        __device__
        TextureInfo getTexture(unsigned int index) {
            return gpuTextureBuffer[index];
        }

    } // namespace Cuda
} // namespace Renderer
