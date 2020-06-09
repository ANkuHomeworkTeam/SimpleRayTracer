#include "tasks.hpp"
#include "sobol.hpp"
#include "cudaRenderer.hpp"
#include "details.hpp"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

namespace Renderer
{

namespace Cuda
{

    inline cudaError_t checkCudaError() {
        auto error = cudaGetLastError();
        if (error != 0)
            cout<<"cudaError:"<<error<<" - "<<cudaGetErrorString(error)<<endl;
        return error;
    }

    bool checkCudaSupport() {
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        if (nDevices == 0) return false;
        return true;
    }

    inline const RenderConfig& getRenderConfig() {
        return renderConfig;
    }

    ErrorCode cudaRender(Vec3** pixels) {
        if (globalEnv == RenderEnv::UNDEFINE) {
            return ErrorCode::NOT_INIT;
        }
        else if (globalEnv != RenderEnv::CUDA) {
            return ErrorCode::ENV_NOT_SUPPORT;
        }

        auto width = getRenderConfig().width;
        auto height = getRenderConfig().height;

        // check device
        cudaDeviceProp device;
        cudaGetDeviceProperties(&device, 0);
        auto maxThreadPerBlock = device.maxThreadsPerBlock;
        // auto sharedMenPerBlockKb = device.sharedMemPerBlock;
        
        Renderer::Cuda::initSobolSequence();
        initRayGenerator();
        //cudaMemcpyToSymbol(gpuSobolSequence, sobolSequence,
        //    sizeof(Vec3)*SOBOL_SEQUENCE_CYCLE);
        // TODO: Compute Bounding box

        Vec3* generatedPixels;
        cudaMallocManaged(&generatedPixels, sizeof(Vec3)*width*height);

        dim3 blockNum(width, height);
        dim3 threadNum(getRenderConfig().sampleNums);

        renderTask<<<blockNum, threadNum>>>(generatedPixels, width, height, getRenderConfig().depth,initTexture(), initMaterial(), initObject(), initVertex(), objectBuffer.size());

        cudaDeviceSynchronize();

        if(checkCudaError()!=0) return ErrorCode::CUDA_ERROR;

        if(renderConfig.gamma) {
            // Wrap optimize
            gammaTask<<<blockNum, 1>>>(generatedPixels, width, height);
            cudaDeviceSynchronize();
        }

        // render result
        *pixels = generatedPixels;

        cout<<"Render Finished in cuda"<<endl;
        return ErrorCode::SUCCESS;
    }

};
};