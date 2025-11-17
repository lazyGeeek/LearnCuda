#pragma once
#ifndef CUDA_APPLICATION_HPP_
#define CUDA_APPLICATION_HPP_

#include <filesystem>
#include <memory>

namespace OpenGL { class Driver; }
namespace UI { class UIManager; }
namespace Window { class GLFW; }

namespace Cuda
{
    class Application
    {
    public:
        Application(const std::filesystem::path& projectPath);
        ~Application();

        Application(const Application& other)             = delete;
        Application(Application&& other)                  = delete;
        Application& operator=(const Application& other)  = delete;
        Application& operator=(const Application&& other) = delete;

        void Run();

        const std::filesystem::path ProjectPath;
        const std::filesystem::path ConfigPath;

    private:
        std::unique_ptr<OpenGL::Driver> m_opengl = nullptr;
        std::unique_ptr<UI::UIManager> m_uiManager = nullptr;
        std::unique_ptr<Window::GLFW> m_window = nullptr;
    };
}

#endif // CUDA_APPLICATION_HPP_
