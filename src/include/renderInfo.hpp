#pragma once

#ifndef __RENDER_INFO_HPP__
#define __RENDER_INFO_HPP__

#include "renderer.hpp"
#include <vector>
namespace Renderer
{
    enum class MaterialType
    {
        LAMBERTAIN,
        PHONG,
        SPECULAR,
        GLASS  
    };

    enum class ObjectType
    {
        Sphere,
        Triangle
    };

    enum class TextureType
    {
        Solid
    };

    
    struct MaterialInfo
    {
        MaterialType    type;
        id_t            id;
        id_t            texture;
        Vec3            v1;
        Vec3            v2;
        Vec3            v3;
        float           f1;
        float           f2;
        float           f3;
    };
    /*
    struct MaterialInfo
    {
        MaterialType    type;
        id_t            id;
        // diffuse or reflect's attenuation
        id_t            texture;
        float           glossy;
        // Phong reflection
        float           phongShininess;
    };
    */

    struct ObjectInfo
    {
        ObjectType      type;
        id_t            id;
        id_t            material;
        Vec3            v1;
        Vec3            v2;
        Vec3            v3;
        float           f1;
        float           f2;
        float           f3;
        int             i1;
        int             i2;
        int             i3;
    };

    struct TextureInfo
    {
        TextureType     type;
        id_t            id;
        Vec3            v1;
    };

    extern RenderEnv globalEnv;

    extern std::vector<Vec3> vertexBuffer;
    extern std::vector<TextureInfo> textureBuffer;
    extern std::vector<MaterialInfo> materialBuffer;
    extern std::vector<ObjectInfo> objectBuffer;

    extern Camera cam;

    extern RenderConfig renderConfig;

}; // namespace Renderer

#endif