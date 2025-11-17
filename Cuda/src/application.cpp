#include "cuda/application.hpp"
#include "cuda/panels/menu.hpp"
#include "open_gl/driver.hpp"
#include "ui/ui_manager.hpp"
#include "ui/widgets/visuals/image.hpp"
#include "window/glfw.hpp"

#include <array>
#include <string>
#include <vector>

namespace Cuda
{
    Application::Application(const std::filesystem::path& projectPath) :
        ProjectPath(projectPath),
        ConfigPath(ProjectPath / "Confings")
    {
        if (!std::filesystem::exists(ConfigPath))
            std::filesystem::create_directory(ConfigPath);

        Window::Settings::WindowSettings settings;
        settings.Title = "Learn CUDA";
        m_window = std::make_unique<Window::GLFW>(settings);

        m_opengl = std::make_unique<OpenGL::Driver>(true);
        
        m_uiManager = std::make_unique<UI::UIManager>(m_window->GetWindow(), ConfigPath / "ImGui.ini", "#version 460");
        m_uiManager->EnableDocking(true);
        m_uiManager->ApplyStyle(UI::Styling::EStyle::ImDarkStyle);
    }

    Application::~Application()
    {
        if (m_uiManager)
            m_uiManager = nullptr;

        if (m_opengl)
            m_opengl = nullptr;

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

            std::shared_ptr<Panels::MenuPanel> menu = canvas->CreatePanel<Panels::MenuPanel>(canvas);
            menu->Resize(300.0f, 500.0f);
        }

        m_window->CloseEvent += [&]()
        {
            canvas->RemoveAllPanels();
        };

        while (m_window && !m_window->ShouldClose())
        {
            m_window->PollEvents();
            
            if (m_uiManager)
                m_uiManager->Render();

            m_window->SwapBuffers();
        }
    }
}
