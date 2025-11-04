#pragma once
#ifndef LEARN_CUDA_PANELS_CUDA_INFO_HPP_
#define LEARN_CUDA_PANELS_CUDA_INFO_HPP_

#include "ui/panels/window_panel.hpp"

namespace LearnCuda::Panels
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

#endif // LEARN_CUDA_PANELS_CUDA_INFO_HPP_
