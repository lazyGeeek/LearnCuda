#pragma once
#ifndef LEARN_CUDA_APPLICATION_HPP_
#define LEARN_CUDA_APPLICATION_HPP_

#include <filesystem>
#include <memory>

namespace Window { class GLFW; }

namespace LearnCuda
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
    };
}

#endif // LEARN_CUDA_APPLICATION_HPP_
