#pragma once
#ifndef CUDA_PANELS_MENU_PANEL_HPP_
#define CUDA_PANELS_MENU_PANEL_HPP_

#include "ui/panels/window_panel.hpp"

namespace UI::Modules { class Canvas; }

namespace Cuda::Panels
{
    class CudaInfoPanel;
    class JuliaFractal;
    class Ripples;
    class ThreadsSync;
    class Raytracing;
    class Heat;

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
        std::shared_ptr<Ripples> m_ripples = nullptr;
        std::shared_ptr<ThreadsSync> m_threadsSync = nullptr;
        std::shared_ptr<Raytracing> m_raytracing = nullptr;
        std::shared_ptr<Heat> m_heat = nullptr;
    };
}

#endif // CUDA_PANELS_MENU_PANEL_HPP_
