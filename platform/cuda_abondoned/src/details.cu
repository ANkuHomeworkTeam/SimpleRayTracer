#include <iostream>
#include <stdio.h>
#include "renderInfo.hpp"
#include "sobol.hpp"
#include <cuda_runtime.h>
namespace Renderer
{
    namespace Cuda
    {   
        Vec3*   initVertex() {
            Vec3* tmp;
            cudaMalloc((void**)&tmp, vertexBuffer.size()*sizeof(Vec3));
            cudaMemcpy(tmp, &vertexBuffer[0],
            vertexBuffer.size()*sizeof(Vec3), cudaMemcpyHostToDevice);
            return tmp;
        }

        TextureInfo* initTexture() {
            TextureInfo* tmp;
            cudaMalloc((void**)&tmp,
            textureBuffer.size()*sizeof(TextureInfo));
            cudaMemcpy(tmp, &textureBuffer[0],
            textureBuffer.size()*sizeof(TextureInfo), cudaMemcpyHostToDevice);
            return tmp;
        }

        MaterialInfo* initMaterial() {
            MaterialInfo* tmp;
            cudaMalloc((void**)&tmp,
            materialBuffer.size()*sizeof(MaterialInfo));
            cudaMemcpy(tmp, &materialBuffer[0],
            materialBuffer.size()*sizeof(MaterialInfo), cudaMemcpyHostToDevice);
            return tmp;
            //cudaMemcpyToSymbol(gpuMaterialBuffer, &tmp, sizeof(MaterialInfo*));
        }

        ObjectInfo* initObject () {
            ObjectInfo* tmp;
            cudaMalloc((void**)&tmp,
            objectBuffer.size()*sizeof(ObjectInfo));
            cudaMemcpy(tmp, &objectBuffer[0],
            objectBuffer.size()*sizeof(ObjectInfo), cudaMemcpyHostToDevice);
            //cudaMemcpyToSymbol(gpuObjectBuffer, &tmp, sizeof(ObjectInfo*));
            return tmp;
        }
    }; // namespace Cuda
    
}; // namespace Renderer
