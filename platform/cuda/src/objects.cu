#include "objects.hpp"
#include "maths/random.hpp"
#include <stdio.h>
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
        __device__
        id_t* gpuLightBuffer;
        __device__
        int lightNum;

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
            id_t* tmpL;
            cudaMalloc((void**)&tmpL, sizeof(id_t)*lightBuffer.size());
            cudaMemcpy(tmpL, &lightBuffer[0], sizeof(id_t)*lightBuffer.size(), cudaMemcpyHostToDevice);
            cudaMemcpyToSymbol(gpuLightBuffer, &tmpL, sizeof(id_t*));
            int lsize = lightBuffer.size();
            cudaMemcpyToSymbol(lightNum, &lsize, sizeof(id_t));
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

        __device__
        LightSampling sampleRandomLight(const Vec3& from, const Vec3& faceDirection) {
            Vec3 samplePoint;
            id_t sampleId;
            id_t lightId = std::floor(getRandom()*float(lightNum));
            for (int i=0; i<lightNum; i++) {
                float discrim;
                sampleId = gpuLightBuffer[(lightId + i)%lightNum];
                ObjectInfo& obj = gpuObjectBuffer[sampleId];
                if (obj.type == ObjectType::Triangle) {
                    float u = getRandom();
                    float v = getRandom() * (1 - u);
                    Vec3 v1 = gpuVertexBuffer[obj.i1];
                    Vec3 v2 = gpuVertexBuffer[obj.i1 + 1];
                    Vec3 v3 = gpuVertexBuffer[obj.i1 + 2];
                    Vec3 e1 = v2 - v1;
                    Vec3 e2 = v3 - v1;
                    samplePoint = v1 + e1 * u + e2 * v;
                    discrim = dot(samplePoint - from, faceDirection);
                }
                else if (obj.type == ObjectType::Sphere) {
                    samplePoint = obj.v1 + getSobolNormalized(threadIdx.x);
                    discrim = dot(samplePoint - from, faceDirection);
                }
                if (discrim > 0) return createLightSampling(sampleId, samplePoint);
            }
            return nullopt;
        }

#   pragma region HIT_TEST

        __device__
        HitRecord hit(id_t object, const Ray& ray, float min = 0.f, float max = FLOAT_INF) {
            ObjectInfo& obj = gpuObjectBuffer[object];
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
                        return createHitRecord(temp, object, hitPoint, normal, obj.material);
                    }
                    temp = (-b + sqrtdiscrim) / a;
                    if (temp < max && temp >= min) {
                        auto hitPoint = ray.at(temp);
                        auto normal = (hitPoint - obj.v1) / obj.f1;
                        if (dot(normal, ray.direction)>0) normal = neg(normal);
                        return createHitRecord(temp, object, hitPoint, normal, obj.material);
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
                return createHitRecord(t, object, ray.at(t), normal, obj.material);
            }
            return nullopt;
        }

        __device__
        HitRecord intersectionTest(const Ray& ray) {
            int size = getObjectNum();
            float closet = FLOAT_INF;
            HitRecord hitRecord = nullopt;
            for (int i = 0; i < size; i++) {
                auto record = hit(i, ray, 0.0001, closet);
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
