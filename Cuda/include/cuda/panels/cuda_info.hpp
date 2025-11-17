#pragma once
#ifndef CUDA_PANELS_CUDA_INFO_HPP_
#define CUDA_PANELS_CUDA_INFO_HPP_

#include "ui/panels/window_panel.hpp"

namespace Cuda::Panels
{
    class CudaInfoPanel : public UI::Panels::WindowPanel
    {
    public:
        CudaInfoPanel();

        CudaInfoPanel(const CudaInfoPanel& other)             = delete;
        CudaInfoPanel(CudaInfoPanel&& other)                  = delete;
        CudaInfoPanel& operator=(const CudaInfoPanel& other)  = delete;
        CudaInfoPanel& operator=(const CudaInfoPanel&& other) = delete;
    };
}

#endif // CUDA_PANELS_CUDA_INFO_HPP_
