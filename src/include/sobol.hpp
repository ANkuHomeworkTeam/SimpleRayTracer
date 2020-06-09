#pragma once
#ifndef __SOBOL_HPP__
#define __SOBOL_HPP__

#include "renderer.hpp"

#define SOBOL_SEQUENCE_CYCLE 1024

namespace Renderer
{
    extern Vec3 sobolSequence[SOBOL_SEQUENCE_CYCLE];
    void initSobolSequence();
};

#endif