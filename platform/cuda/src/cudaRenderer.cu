#include "cudaRenderer.hpp"
#include "tasks.hpp"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
using namespace Renderer;

bool Cuda::checkCudaSupport() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) return false;
    return true;
}

ErrorCode Cuda::cudaRender(Vec3** pixels) {
    if (globalEnv == RenderEnv::UNDEFINE) {
        return ErrorCode::NOT_INIT;
    }
    else if (globalEnv != RenderEnv::CUDA) {
        return ErrorCode::ENV_NOT_SUPPORT;
    }

    int height = renderConfig.height;
    int width  = renderConfig.width;

    // check device
    cudaDeviceProp device;
    cudaGetDeviceProperties(&device, 0);
    auto maxThreadPerBlock = device.maxThreadsPerBlock;
    auto sharedMenPerBlockKb = device.sharedMemPerBlock;

    // check end

    TextureInfo*    gpuTextureBuffer;
    MaterialInfo*   gpuMaterialBuffer;
    ObjectInfo*     gpuObjectBuffer;

    cudaMalloc((void**)&gpuTextureBuffer,
        textureBuffer.size()*sizeof(TextureInfo));
    cudaMalloc((void**)&gpuMaterialBuffer,
        materialBuffer.size()*sizeof(MaterialInfo));
    cudaMalloc((void**)&gpuObjectBuffer,
        objectBuffer.size()*sizeof(ObjectInfo));

    cudaMemcpy(gpuTextureBuffer, &textureBuffer[0],
        textureBuffer.size()*sizeof(TextureInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuMaterialBuffer, &materialBuffer[0],
        materialBuffer.size()*sizeof(MaterialInfo), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuObjectBuffer, &objectBuffer[0],
        objectBuffer.size()*sizeof(ObjectInfo), cudaMemcpyHostToDevice);

    Vec3* generatedPixels;
    cudaMallocManaged(&generatedPixels, sizeof(Vec3)*width*height);

    dim3 blockNum(height, width);
    dim3 threadNum(256);
    renderTask<<<blockNum, threadNum>>>(generatedPixels, width, height);

    cudaDeviceSynchronize();

    *pixels = generatedPixels;


    return ErrorCode::SUCCESS;
}

