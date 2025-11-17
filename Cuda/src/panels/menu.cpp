#include "cuda/panels/menu.hpp"
#include "cuda/panels/cuda_info.hpp"
#include "cuda/panels/julia_fractal.hpp"
#include "cuda/panels/ripples.hpp"
#include "cuda/panels/threads_sync.hpp"
#include "ui/modules/canvas.hpp"
#include "ui/widgets/buttons/button.hpp"

namespace Cuda::Panels
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

        m_ripples = m_canvas->CreatePanel<Panels::Ripples>();
        m_ripples->SetAutoSize(true);
        m_ripples->Resize(1100.0f, 700.0f);
        m_ripples->SetPosition(50.0f, 50.0f);
        m_ripples->SetOpened(false);

        m_threadsSync = m_canvas->CreatePanel<Panels::ThreadsSync>();
        m_threadsSync->SetAutoSize(true);
        m_threadsSync->Resize(1100.0f, 700.0f);
        m_threadsSync->SetPosition(50.0f, 50.0f);
        m_threadsSync->SetOpened(false);

        UIButtonPtr cudaInfoButton = CreateWidget<UIButton>("Show GPU Info");
        cudaInfoButton->Separate(true);
        UIButtonPtr juliaFractalButton = CreateWidget<UIButton>("Julia Fractal");
        UIButtonPtr wavesButton = CreateWidget<UIButton>("Ripples");
        UIButtonPtr threadsSyncButton = CreateWidget<UIButton>("Threads Sync");

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
            if (m_ripples) m_ripples->SetOpened(true);
        };

        threadsSyncButton->ClickedEvent += [&]()
        {
            if (m_threadsSync) m_threadsSync->SetOpened(true);
        };
    }
}
