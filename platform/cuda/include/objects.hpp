#pragma once
#ifndef __CUDA_OBJECTS_HPP__
#define __CUDA_OBJECTS_HPP__

#include "renderInfo.hpp"
#include "details.hpp"
#include <cuda_runtime.h>
#include "maths/float.hpp"
#include <thrust/optional.h>
#include <thrust/functional.h>

namespace Renderer
{
    namespace Cuda
    {
        void initObjects();
        __device__
        const ObjectInfo& getObject(unsigned int index);
        __device__
        const Vec3& getVertex(unsigned int index);
        __device__
        int getObjectNum();

        struct HitRecordBase
        {
            float t;
            Vec3 hitPoint;
            Vec3 normal;
            id_t material;
            __device__
            HitRecordBase():
                t(FLOAT_INF), hitPoint(), normal(), material(0)
            {}
            __device__
            HitRecordBase(float t, const Vec3& hitPoint, const Vec3& normal, id_t material):
                t(t), hitPoint(hitPoint), normal(normal), material(material)
            {}
        };
        using HitRecord = thrust::optional<HitRecordBase>;
        template<typename ...Args>
        __device__ HitRecord createHitRecord(Args&& ...args) {
            return thrust::make_optional<HitRecordBase>(args...);
        }
        __device__
        HitRecord intersectionTest(const Ray& ray);
    } // namespace Cuda
} // namespace Renderer


#endif