#include "sobol.hpp"
#include "maths/random.hpp"
#include "maths/operation.hpp"
#include <curand_kernel.h>
#include <time.h>

namespace Renderer
{
    namespace Cuda
    {
        __constant__
        Vec3 gpuSobolSequence[SOBOL_SEQUENCE_CYCLE];
        __constant__
        Vec3 gpuSobolSequenceNormalized[SOBOL_SEQUENCE_CYCLE];

        __device__
        curandState *cState;

        __global__
        void initState(unsigned long seed, curandState* s) {
            int id = threadIdx.x;
            curand_init(seed, id, 0, &s[id]);
        }

        void initRandom(int threadNum) {
#       pragma region INIT_SOBOL
            cudaMemcpyToSymbol(gpuSobolSequence, sobolSequence,
            sizeof(Vec3)*SOBOL_SEQUENCE_CYCLE);
            Vec3* normalized = new Vec3[SOBOL_SEQUENCE_CYCLE];
            for(int i=0; i<SOBOL_SEQUENCE_CYCLE; i++) {
                auto v = sobolSequence[i] - 0.5;
                if (length(v) < 0.001f) v = { 1, 1, 1 };
                normalized[i] = normalize(v);
            }
            cudaMemcpyToSymbol(gpuSobolSequenceNormalized, normalized,
            sizeof(Vec3)*SOBOL_SEQUENCE_CYCLE);
#       pragma endregion INIT_SOBOL

#       pragma region INIT_RANDOM
            curandState* tmp;
            cudaMalloc((void**)&tmp, sizeof(curandState)*threadNum);
            initState<<<1, threadNum>>>(time(NULL), tmp);
            cudaMemcpyToSymbol(cState, &tmp, sizeof(curandState*));
#       pragma endregion
        }

        __device__
        Vec3 getSobol(int index) {
            return gpuSobolSequence[index%SOBOL_SEQUENCE_CYCLE];
        }
        __device__
        Vec3 getSobolNormalized(int index) {
            __shared__
            static int counter;
            atomicAdd(&counter, 1);
            return gpuSobolSequenceNormalized[(index+counter)%SOBOL_SEQUENCE_CYCLE];
        }
        __device__
        float getRandom() {
            return curand_uniform(&cState[threadIdx.x]);
        }

        __device__
        Vec3 getRandomNormalizedVec3() {
            return normalize({getRandom() - 0.5f, getRandom() - 0.5f, getRandom() - 0.5f});
        }
    } // namespace Cuda
} // namespace Renderer
