#include "cudaRenderer.hpp"
#include "maths/random.hpp"
#include "renderInfo.hpp"
#include "renderer.hpp"
#include "tasks.hpp"
#include "objects.hpp"
#include "BSDFs.hpp"
#include "details.hpp"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

namespace Renderer
{
    namespace Cuda
    {
        inline
        cudaError_t checkCudaError() {
            auto error = cudaGetLastError();
            if (error != 0)
                cerr<<" - Cuda Error: "<<error<<endl
                    <<"   "<<cudaGetErrorString(error)<<endl;
            return error;
        }
        bool checkCudaSupport() {
            int nDevices;
            cudaGetDeviceCount(&nDevices);
            if (nDevices == 0) return false;
            return true;
        }

        ErrorCode cudaRender(Vec3** pixels) {
#       pragma region CHECK_ENV_AND_CUDA
            if (globalEnv == RenderEnv::UNDEFINE) {
                return ErrorCode::NOT_INIT;
            }
            else if (globalEnv != RenderEnv::CUDA) {
                return ErrorCode::ENV_NOT_SUPPORT;
            }
#       pragma endregion

        unsigned int width = renderConfig.width;
        unsigned int height = renderConfig.height;
        unsigned int sumpleNums = renderConfig.sampleNums;

#       pragma region INIT
        initRandom(sumpleNums);
        initRenderConfig();
        initRayGenerator();
        initTexture();
        initMaterial();
        initObjects();
#       pragma endregion

#       pragma region CONFIG_RENDER_KERNEL
        dim3 blockInfo{width, height};
        dim3 threadInfo{sumpleNums};
#       pragma endregion

#       pragma region RENDER
        Vec3* generatedPixels;
        cudaMallocManaged(&generatedPixels, sizeof(Vec3)*width*height);

        renderTask<<<blockInfo, threadInfo>>>(generatedPixels);

        cudaDeviceSynchronize();

        if(checkCudaError()!=0) return ErrorCode::CUDA_ERROR;
#       pragma endregion

        *pixels = generatedPixels;
        cout<<"Render Finished in cuda"<<endl;
        return ErrorCode::SUCCESS;
        }
    }
}