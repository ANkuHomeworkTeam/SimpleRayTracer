#include "objects.hpp"
#include <iostream>
namespace Renderer
{
    using namespace thrust;
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
        const ObjectInfo& getObject(unsigned int index) {
            return gpuObjectBuffer[index];
        }
        __device__
        const Vec3& getVertex(unsigned int index) {
            return gpuVertexBuffer[index];
        }

        __device__
        int getObjectNum() {
            return objectNum;
        }

#   pragma region HIT_TEST

        __device__
        HitRecord hit(const ObjectInfo& obj, const Ray& ray, float min = 0.f, float max = FLOAT_INF) {
            if (obj.type == ObjectType::Sphere) {
                Vec3 oc = ray.origin - obj.v1;
                float a = dot(ray.direction, ray.direction);
                float b = dot(oc, ray.direction);
                float c = dot(oc, oc) - obj.f1*obj.f1;
                float discriminant = b*b - a*c;
                float sqrtdiscrim = std::sqrt(discriminant);
                if (discriminant > 0) {
                    float temp = (-b - sqrtdiscrim) / a;
                    if (temp < max && temp >= min) {
                        auto hitPoint = ray.at(temp);
                        auto normal = (hitPoint - obj.v1) / obj.f1;
                        if (dot(normal, ray.direction)>0) normal = neg(normal);
                        return createHitRecord(temp, hitPoint, normal, obj.material);
                    }
                    temp = (-b + sqrtdiscrim) / a;
                    if (temp < max && temp >= min) {
                        auto hitPoint = ray.at(temp);
                        auto normal = (hitPoint - obj.v1) / obj.f1;
                        if (dot(normal, ray.direction)>0) normal = neg(normal);
                        return createHitRecord(temp, hitPoint, normal, obj.material);
                    }
                }
                return nullopt;
            }
            else if (obj.type == ObjectType::Triangle) {
                auto castVertex = [&obj] (int i) -> const Vec3& {
                    return getVertex(obj.i1 + i);
                };
                auto e1 = castVertex(1) - castVertex(0);
                auto e2 = castVertex(2) - castVertex(0);
                auto normal = normalize(cross(e1, e2));
                auto P = cross(ray.direction, e2);
                float det = dot(e1, P);
                Vec3 T;
                if (det > 0)
                    T = ray.origin - castVertex(0);
                else {
                    T = castVertex(0) - ray.origin;
                    det = -det;
                }
                if (det < 0.000001f) return nullopt;
                float u, v, t;
                u = dot(T, P);
                if (u > det || u < 0.f) {
                    return nullopt;
                }
                Vec3 Q = cross(T, e1);
                v = dot(ray.direction, Q);
                if (v < 0.f || v + u > det) {
                    return nullopt;
                }
                t = dot(e2, Q);
                float fInvDet = 1.f / det;
                t *= fInvDet;
                if (t >= max || t < min) return nullopt;
                if (dot(normal, ray.direction) > 0) {
                    normal = -normal;
                }
                return createHitRecord(t, ray.at(t), normal, obj.material);
            }
            return nullopt;
        }

        __device__
        HitRecord intersectionTest(const Ray& ray) {
            int size = getObjectNum();
            float closet = FLOAT_INF;
            HitRecord hitRecord = nullopt;
            for (int i = 0; i < size; i++) {
                auto record = hit(getObject(i), ray, 0.0001, closet);
                if (record) {
                    closet = record->t;
                    hitRecord = record;
                }
            }
            return hitRecord;
        }
#   pragma endregion
    } // namespace Cuda
} // namespace Renderer
