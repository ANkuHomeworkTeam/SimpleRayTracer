#include "renderer.hpp"
#include <iostream>

using namespace std;
using namespace Renderer;

int main() {
    init(RenderEnv::CUDA);
    Vec3* p;
    setRenderConfig(1920, 1080, true);
    render(&p);
    for(int i=0;i<100;i++)
    {
        cout<<"["<<p[i].x<<", "<<p[i].y<<", "<<p[i].z<<" ]"<<endl;
    }
    return 0;
}