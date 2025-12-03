#pragma once
#ifndef CUDA_PANELS_RIPPLES_HPP_
#define CUDA_PANELS_RIPPLES_HPP_

#include "ui/panels/window_panel.hpp"

#include <atomic>
#include <thread>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }
namespace UI::Widgets::Texts { class Text; }

namespace Cuda::Panels
{
    class Ripples : public UI::Panels::WindowPanel
    {
    public:
        Ripples();
        ~Ripples() override;

        Ripples(const Ripples& other)             = delete;
        Ripples(Ripples&& other)                  = delete;
        Ripples& operator=(const Ripples& other)  = delete;
        Ripples& operator=(const Ripples&& other) = delete;

    protected:
        virtual void DrawImpl() override;

    private:
        void calculateRipplesOnCPU(int64_t ticks);
        void calculateRipplesOnGPU(int64_t ticks);

        int ripplesCPU(int x, int y, float ticks);

        friend __global__ void ripplesGpu(uint8_t* buffer, float ticks, Ripples& ripples);

        const size_t IMAGE_WIDTH = 512;
        const size_t IMAGE_HEIGHT = 512;
        const size_t THREADS_COUNT = 16;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 4;
        
        std::vector<uint8_t> m_cpuImageBuffer;
        std::vector<uint8_t> m_gpuImageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_cpuTexture = nullptr;
        std::unique_ptr<OpenGL::Resources::Texture> m_gpuTexture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_cpuCalculationTimeText = nullptr;
        std::shared_ptr<UI::Widgets::Texts::Text> m_gpuCalculationTimeText = nullptr;
        
        std::atomic<bool> m_isCPUCalculationRunning { true };
        std::atomic<bool> m_isGPUCalculationRunning { true };

        std::thread m_cpuCalculationThread;
        std::thread m_gpuCalculationThread;
    };
}

#endif // CUDA_PANELS_RIPPLES_HPP_
