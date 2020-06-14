#include "renderer.hpp"
#include <iostream>
#include "svpng/svpng.hpp"
#include <cstdio>

using namespace std;
using namespace Renderer;

void outputPNG(Vec3* pixels, int width, int height, const char* fileName) {
    unsigned char* png = new unsigned char[height*width*3];
    auto p = png;
    for(int i = 0; i < height*width; i++) {
        Vec3& pixel = pixels[i];
        *p++ = (unsigned char)(pixel.r * 255.f);
        *p++ = (unsigned char)(pixel.g * 255.f);
        *p++ = (unsigned char)(pixel.b * 255.f);
    }
    FILE *fp = fopen(fileName, "wb");
    svpng(fp, width, height, png, false);
    fclose(fp);
    delete[] png;
}

int main() {
    int height = 960;
    int width = 960;

    init(RenderEnv::CUDA);
    Vec3* pixels;
    setRenderConfig(width, height, true);
    setCamera(40.f, 1.f, 10.f, 0.0f, {278.f, 278.f, -750.f}, {278.f, 278.f, 0});

    auto whiteT = Texture::createSolid({ .73, .73, .73 });
    auto redT   = Texture::createSolid({ .2f, .05f, .05f });
    auto greenT = Texture::createSolid({ .12f, .45f, .15f });
    auto whiteW = Material::createLambertain(whiteT);
    auto redW   = Material::createLambertain(redT);
    auto greenW = Material::createLambertain(greenT);
    auto mirror = Material::createSpecular(0, 0.f);

    auto light  = Material::createEmitted(0, 1.0, 0, 0);

    Object::createTriangle({555, 0, 555}, {0, 0, 555}, {0, 0, 0}, whiteW);
    Object::createTriangle({555, 0, 555}, {555, 0, 0}, {0, 0, 0}, whiteW);

    Object::createTriangle({555, 0, 555},{555, 555, 555},{0, 0, 555}, whiteW);
    Object::createTriangle({0, 555, 555},{0, 0, 555}, {555, 555, 555}, whiteW);

    Object::createTriangle({555, 555, 555}, {0, 555, 555}, {0, 555, 0}, whiteW);
    Object::createTriangle({555, 555, 555}, {555, 555, 0}, {0, 555, 0}, whiteW);

    Object::createTriangle({555, 555, 555}, {555, 0, 555}, {555, 0, 0}, redW);
    Object::createTriangle({555, 555, 555}, {555, 555, 0}, {555, 0, 0}, redW);

    Object::createTriangle({0, 555, 555}, {0, 0, 555}, {0, 0, 0}, greenW);
    Object::createTriangle({0, 555, 555}, {0, 555, 0}, {0, 0, 0}, greenW);

    Object::createTriangle({378, 554, 378}, {178, 554, 378}, {178, 554, 178}, light);
    Object::createTriangle({378, 554, 378}, {378, 554, 178}, {178, 554, 178}, light);
    
    Object::createSphere({278, 100, 278}, 100, mirror);
    render(&pixels);
    
    outputPNG(pixels, width, height, "../cudaResult.png");

    return 0;
}