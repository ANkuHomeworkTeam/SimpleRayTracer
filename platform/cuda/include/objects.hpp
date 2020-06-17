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
            id_t object;
            Vec3 hitPoint;
            Vec3 normal;
            id_t material;
            __device__
            HitRecordBase():
                t(FLOAT_INF), object(0) ,hitPoint(), normal(), material(0)
            {}
            __device__
            HitRecordBase(float t, id_t object, const Vec3& hitPoint, const Vec3& normal, id_t material):
                t(t), object(object), hitPoint(hitPoint), normal(normal), material(material)
            {}
        };
        using HitRecord = thrust::optional<HitRecordBase>;
        template<typename ...Args>
        __device__ HitRecord createHitRecord(Args&& ...args) {
            return thrust::make_optional<HitRecordBase>(args...);
        }
        __device__
        HitRecord intersectionTest(const Ray& ray);
        struct LightSamplingBase {
            id_t objId;
            Vec3 samplePoint;
            __device__
            LightSamplingBase(id_t objId, const Vec3& samplePoint):
                objId(objId), samplePoint(samplePoint)
            {}
        };
        using LightSampling = thrust::optional<LightSamplingBase>;
        __device__ inline
        LightSampling createLightSampling(id_t objId, const Vec3& samplePoint) {
            return thrust::make_optional<LightSamplingBase>(objId, samplePoint);
        }

        __device__
        LightSampling sampleRandomLight(const Vec3& from, const Vec3& faceDirection);
    } // namespace Cuda
} // namespace Renderer


#endif