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
    int height = 540;
    int width = 960;

    init(RenderEnv::CUDA);
    Vec3* pixels;
    setRenderConfig(width, height, true);
    setCamera(40.f, 1.f, 0.f, 10.f, {278.f, 278.f, -750.f}, {278.f, 278.f, 0});

    Texture::createSolid({0.5, 0.4, 0.3});
    Material::createLambertain(1);
    Object::createTriangle({555, 0, 0},{555, 555, 0},{0, 0, 0}, 1);
    render(&pixels);
    
    outputPNG(pixels, width, height, "../cudaResult.png");

    return 0;
}