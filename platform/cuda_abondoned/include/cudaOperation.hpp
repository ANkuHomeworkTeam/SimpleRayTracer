#pragma once
#ifndef __CUDA_OPERATION_HPP__
#define __CUDA_OPERATION_HPP__

#include <cuda_runtime.h>
#include <cmath>

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
        __host__ __device__ inline
        Vec3 mul(const Vec3& v, float f) {
            return { v.x * f, v.y * f, v.z * f };
        }
        __host__ __device__ inline
        Vec3 div(const Vec3& v1, const Vec3& v2) {
            return { v1.x / v2.x, v1.y / v2.y, v1.z / v2.z };
        }
        __host__ __device__ inline
        Vec3 div(const Vec3& v, float f) {
            return { v.x / f, v.y / f, v.z / f };
        }

        __host__ __device__ inline
        Vec3 sqrt(const Vec3& v) {
            return { std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z) };
        }

        __host__ __device__ inline
        Vec3 pow(const Vec3& v, float s) {
            return { std::pow(v.x, s), std::pow(v.y, s), std::pow(v.z, s) };
        }

        __host__ __device__ inline
        float dot(const Vec3& v1, const Vec3& v2) {
            return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
        }

        __host__ __device__ inline
        float length(const Vec3& v) {
            return std::sqrt(dot(v,v));
        }

        __host__ __device__ inline
        float cos(const Vec3& v1, const Vec3& v2) {
            return dot(v1, v2) / (length(v1)*length(v2));
        }

        __host__ __device__ inline
        float radians(float angle) {
            return 3.1415926f * angle / 180.f;
        }

        __host__ __device__ inline
        Vec3 normalize(const Vec3& v) {
            return div(v, length(v));
        }

        __host__ __device__ inline
        Vec3 cross(const Vec3& v1, const Vec3& v2) {
            const float& Xu = v1.x;
            const float& Yu = v1.y;
            const float& Zu = v1.z;
            const float& Xv = v2.x;
            const float& Yv = v2.y;
            const float& Zv = v2.z;
            return { Yu*Zv - Zu*Yv, Zu*Xv - Xu*Zv, Xu*Yv - Yu*Xv };
        }
    }
};

#endif