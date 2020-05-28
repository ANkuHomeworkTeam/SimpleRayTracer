#include "renderer.hpp"
#include "renderInfo.hpp"
#include <vector>

using namespace Renderer;
using namespace std;

RenderEnv globalEnv = RenderEnv::UNDEFINE;

std::vector<Vec3> vertexBuffer;
std::vector<TextureInfo> textureBuffer;
std::vector<MaterialInfo> materialBuffer;
std::vector<ObjectInfo> objectBuffer;

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
    Texture::createSolid({1, 1, 1});
    Material::createLambertain(0);
    return ErrorCode::SUCCESS;
}

id_t addVertex(const Vec3& p) {
    vertexBuffer.push_back(p);
    return vertexBuffer.size() - 1;
}

id_t Texture::createSolid(const Vec3& rgb) {
    TextureInfo temp;
    int index = textureBuffer.size();
    temp.type = TextureType::Solid;
    temp.id   = index;
    temp.v1   = rgb;
    textureBuffer.push_back(temp);
    return index;
}

id_t Material::createLambertain(id_t texture) {
    MaterialInfo temp;
    int index    = materialBuffer.size();
    temp.type    = MaterialType::LAMBERTAIN;
    temp.id      = index;
    temp.texture = texture;
    materialBuffer.push_back(temp);
    return index;
}

id_t Object::createSphere(const Vec3& position, float radius, id_t material) {
    ObjectInfo temp;
    int index     = objectBuffer.size();
    temp.type     = ObjectType::Sphere;
    temp.id       = index;
    temp.material = material;
    temp.v1       = position;
    temp.f1       = radius;
    objectBuffer.push_back(temp);
    return index;
}

id_t Object::createTriangle(const Vec3& p1, const Vec3& p2, const Vec3& p3, id_t material) {
    ObjectInfo temp;
    int index     = objectBuffer.size();
    temp.type     = ObjectType::Triangle;
    temp.id       = index;
    temp.material = material;
    temp.i1       = addVertex(p1);
    addVertex(p2);
    addVertex(p3);
    return index;
}