#include "learn_cuda/panels/menu.hpp"
#include "learn_cuda/panels/cuda_info.hpp"
#include "learn_cuda/panels/julia_fractal.hpp"
#include "learn_cuda/panels/waves.hpp"
#include "ui/modules/canvas.hpp"
#include "ui/widgets/buttons/button.hpp"

namespace LearnCuda::Panels
{
    using UIButton = UI::Widgets::Buttons::Button;
    using UIButtonPtr = std::shared_ptr<UIButton>;

    MenuPanel::MenuPanel(std::shared_ptr<UI::Modules::Canvas> canvas) :
        UI::Panels::WindowPanel("Menu"),
        m_canvas { canvas }
    {
        m_cudaInfo = m_canvas->CreatePanel<CudaInfoPanel>();
        m_cudaInfo->Resize(300.0f, 500.0f);
        m_cudaInfo->SetOpened(false);

        m_juliaFractal = m_canvas->CreatePanel<Panels::JuliaFractal>();
        m_juliaFractal->SetAutoSize(true);
        m_juliaFractal->Resize(1100.0f, 700.0f);
        m_juliaFractal->SetPosition(50.0f, 50.0f);
        m_juliaFractal->SetOpened(false);

        m_waves = m_canvas->CreatePanel<Panels::Waves>();
        m_waves->SetAutoSize(true);
        m_waves->Resize(1100.0f, 700.0f);
        m_waves->SetPosition(50.0f, 50.0f);
        m_waves->SetOpened(false);

        UIButtonPtr cudaInfoButton = CreateWidget<UIButton>("Show GPU Info");
        cudaInfoButton->Separate(true);
        UIButtonPtr juliaFractalButton = CreateWidget<UIButton>("Julia Fractal");
        UIButtonPtr wavesButton = CreateWidget<UIButton>("Waves");

        cudaInfoButton->ClickedEvent += [&]()
        {
            if (m_cudaInfo) m_cudaInfo->SetOpened(true);
        };

        juliaFractalButton->ClickedEvent += [&]()
        {
            if (m_juliaFractal) m_juliaFractal->SetOpened(true);
        };

        wavesButton->ClickedEvent += [&]()
        {
            if (m_waves) m_waves->SetOpened(true);
        };
    }
}
