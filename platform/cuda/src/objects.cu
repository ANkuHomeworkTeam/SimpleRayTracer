#include "objects.hpp"
#include <iostream>
namespace Renderer
{
    namespace Cuda
    {
        __device__
        ObjectInfo* gpuObjectBuffer;
        __device__
        Vec3* gpuVertexBuffer;
        __device__
        int objectNum;

        void initObjects() {
            Vec3* tmpV;
            cudaMalloc((void**)&tmpV, sizeof(Vec3)*vertexBuffer.size());
            cudaMemcpy(tmpV, &vertexBuffer[0], sizeof(Vec3)*vertexBuffer.size(), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(gpuVertexBuffer, &tmpV, sizeof(Vec3*));
            ObjectInfo* tmpO;
            cudaMalloc((void**)&tmpO, sizeof(ObjectInfo)*objectBuffer.size());
            cudaMemcpy(tmpO, &objectBuffer[0], sizeof(ObjectInfo)*objectBuffer.size(), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(gpuObjectBuffer, &tmpO, sizeof(ObjectInfo*));
            int size = objectBuffer.size();
            cudaMemcpyToSymbol(objectNum, &size, sizeof(int));
            // cudaMemcpy
        }

        __device__
        ObjectInfo getObject(unsigned int index) {
            return gpuObjectBuffer[index];
        }
        __device__
        Vec3 getVertex(unsigned int index) {
            return gpuVertexBuffer[index];
        }

        __device__
        int getObjectNum() {
            return objectNum;
        }
    } // namespace Cuda
} // namespace Renderer
