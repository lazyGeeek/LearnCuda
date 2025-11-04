#pragma once
#ifndef LEARN_CUDA_PANELS_MENU_PANEL_HPP_
#define LEARN_CUDA_PANELS_MENU_PANEL_HPP_

#include "ui/panels/window_panel.hpp"

namespace UI::Modules { class Canvas; }

namespace LearnCuda::Panels
{
    class CudaInfoPanel;
    class JuliaFractal;

    class MenuPanel : public UI::Panels::WindowPanel
    {
    public:
        MenuPanel(std::shared_ptr<UI::Modules::Canvas> canvas);

        MenuPanel(const MenuPanel& other)             = delete;
        MenuPanel(MenuPanel&& other)                  = delete;
        MenuPanel& operator=(const MenuPanel& other)  = delete;
        MenuPanel& operator=(const MenuPanel&& other) = delete;

    private:
        std::shared_ptr<UI::Modules::Canvas> m_canvas = nullptr;

        std::shared_ptr<CudaInfoPanel> m_cudaInfo = nullptr;
        std::shared_ptr<JuliaFractal> m_juliaFractal = nullptr;
    };
}

#endif // LEARN_CUDA_PANELS_MENU_PANEL_HPP_
