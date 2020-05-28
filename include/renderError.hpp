#pragma once

#ifndef __RENDER_ERROR_HPP__
#define __RENDER_ERROR_HPP__

namespace Renderer
{
    using ErrorDetail = char*;

    enum class ErrorCode
    {
        SUCCESS,
        PLATFORM_NOT_SUPPORT_CUDA,
        ENV_NOT_SUPPORT,
        NOT_INIT
    };

    struct RenderError
    {
        ErrorCode code;
        ErrorDetail detail;
    };
    
};


#endif