#include "renderer.hpp"
#include "renderInfo.hpp"
#include <vector>

using namespace Renderer;
using namespace std;

RenderEnv globalEnv = RenderEnv::UNDEFINE;


bool checkCudaSupport();

ErrorCode init(RenderEnv env) {
    switch (env)
    {
    case RenderEnv::CPU:
        globalEnv = env;
        break;
    case RenderEnv::CPU_OPTIMIZED:
        globalEnv = env;
        break;
    case RenderEnv::CUDA:
        if(checkCudaSupport()) {
            globalEnv = env;
        }
        else {
            return ErrorCode::PLATFORM_NOT_SUPPORT_CUDA;
        }
        break;
    case RenderEnv::UNDEFINE:
        return ErrorCode::ENV_NOT_SUPPORT;
    default:
        return ErrorCode::ENV_NOT_SUPPORT;
    }
    return ErrorCode::SUCCESS;
}
