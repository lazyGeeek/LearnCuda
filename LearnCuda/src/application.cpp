#include "learn_cuda/application.hpp"
#include "learn_cuda/panels/cuda_info.hpp"
#include "ui/ui_manager.hpp"
#include "window/glfw.hpp"

#include <string>
#include <vector>
#include <iostream>

namespace LearnCuda
{
    Application::Application(const std::filesystem::path& projectPath) :
        ProjectPath(projectPath),
        ConfigPath(ProjectPath / "Confings")
    {
        Window::Settings::WindowSettings settings;
        settings.Title = "Learn CUDA";
        m_window = std::make_unique<Window::GLFW>(settings);

        if (!std::filesystem::exists(ConfigPath))
            std::filesystem::create_directory(ConfigPath);
        
        m_uiManager = std::make_unique<UI::UIManager>(m_window->GetWindow(), ConfigPath / "ImGui.ini", "#version 460");
        m_uiManager->EnableDocking(true);
        m_uiManager->ApplyStyle(UI::Styling::EStyle::ImDarkStyle);
    }

    Application::~Application()
    {
        if (m_uiManager)
            m_uiManager = nullptr;

        if (m_window)
            m_window = nullptr;
    }

    void Application::Run()
    {
        std::shared_ptr<UI::Modules::Canvas> canvas = std::make_shared<UI::Modules::Canvas>();
        if (canvas && m_uiManager)
        {
            canvas->SetDockspace(true);
            m_uiManager->SetCanvas(canvas);

            std::shared_ptr<Panels::CudaInfoPanel> profiler = canvas->AddPanel<Panels::CudaInfoPanel>();
            profiler->Resize(300.0f, 500.0f);
        }

        while (m_window && !m_window->ShouldClose())
        {
            m_window->PollEvents();
            
            if (m_uiManager)
                m_uiManager->Render();

            m_window->SwapBuffers();
        }
    }
}
