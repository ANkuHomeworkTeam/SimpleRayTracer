#pragma once

#ifndef __RENDER_INFO_HPP__
#define __RENDER_INFO_HPP__

#include "renderer.hpp"
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

}; // namespace Renderer

#endif