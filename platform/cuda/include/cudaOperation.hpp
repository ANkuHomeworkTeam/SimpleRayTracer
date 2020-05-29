#pragma once
#ifndef __CUDA_OPERATION_HPP__
#define __CUDA_OPERATION_HPP__

#include <cuda_runtime.h>

namespace Renderer {
    namespace Cuda {
        __host__ __device__ inline
        Vec3 neg(const Vec3& v) {
            return { -v.x, -v.y, -v.z };
        }
        __host__ __device__ inline
        Vec3 add(const Vec3& v1, const Vec3& v2) {
            return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
        }
        __host__ __device__ inline
        Vec3 add(const Vec3& v, float f) {
            return { v.x + f, v.y + f, v.z + f };
        }
        __host__ __device__ inline
        Vec3 sub(const Vec3& v1, const Vec3& v2) {
            return { v1.x - v2.x, v1.y - v2.y, v1.z -v2.z };
        }
        __host__ __device__ inline
        Vec3 sub(const Vec3& v, float f) {
            return add(v, -f);
        }
        __host__ __device__ inline
        Vec3 mul(const Vec3& v1, const Vec3& v2) {
            return { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
        }
    }
};

#endif