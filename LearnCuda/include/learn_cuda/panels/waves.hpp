#pragma once
#ifndef LEARN_CUDA_PANELS_WAVES_HPP_
#define LEARN_CUDA_PANELS_WAVES_HPP_

#include "ui/panels/window_panel.hpp"

#include <thread>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }
namespace UI::Widgets::Texts { class Text; }

namespace LearnCuda::Panels
{
    class Waves : public UI::Panels::WindowPanel
    {
    public:
        Waves();
        ~Waves() override;

        Waves(const Waves& other)             = delete;
        Waves(Waves&& other)                  = delete;
        Waves& operator=(const Waves& other)  = delete;
        Waves& operator=(const Waves&& other) = delete;

    protected:
        virtual void DrawImpl() override;

    private:
        void calculateWavesOnCPU(int64_t ticks);
        void calculateWavesOnGPU(int64_t ticks);

        int wavesCPU(int x, int y, float ticks);

        __device__ int wavesGPU(int x, int y, float ticks);
        friend __global__ void gpuStartCalculation(uint8_t* buffer, float ticks, Waves& waves);

        const size_t IMAGE_WIDTH = 500.0f;
        const size_t IMAGE_HEIGHT = 500.0f;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3;
        
        std::vector<uint8_t> m_cpuImageBuffer;
        std::vector<uint8_t> m_gpuImageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_cpuTexture = nullptr;
        std::unique_ptr<OpenGL::Resources::Texture> m_gpuTexture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_cpuCalculationTimeText = nullptr;
        std::shared_ptr<UI::Widgets::Texts::Text> m_gpuCalculationTimeText = nullptr;
        
        bool m_isCPUCalculationRunning = true;
        bool m_isGPUCalculationRunning = true;

        std::thread m_cpuCalculationThread;
        std::thread m_gpuCalculationThread;
    };
}

#endif // LEARN_CUDA_PANELS_WAVES_HPP_
