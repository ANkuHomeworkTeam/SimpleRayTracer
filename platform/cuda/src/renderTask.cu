#include "tasks.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include "details.hpp"
#include "objects.hpp"
#include <thrust/optional.h>
#include <stdio.h>
#include "maths/random.hpp"
#include "maths/operation.hpp"

using namespace thrust;

namespace Renderer
{
    namespace Cuda
    {
        __global__
        void renderTask(Vec3* pixels) {
            //auto v = getSobolNormalized(threadIdx.x);
            //printf("[ %f, %f, %f ]\n", v.x, v.y, v.z);
        }
    } // namespace Cuda
} // namespace Renderer
