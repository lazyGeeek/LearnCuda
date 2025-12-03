#pragma once
#ifndef CUDA_PANELS_JULIA_FRACTAL_HPP_
#define CUDA_PANELS_JULIA_FRACTAL_HPP_

#include "ui/panels/window_panel.hpp"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }

namespace UI::Widgets::Drags { template<typename T> class SingleDrag; }

namespace UI::Widgets::Inputs { template<typename T, size_t Size> class MultipleNumbersInput; }
namespace UI::Widgets::Inputs { template<typename T> class SingleNumberInput; }

namespace UI::Widgets::Texts { class Text; }

namespace Cuda::Panels
{
    class JuliaFractal : public UI::Panels::WindowPanel
    {
    public:
        JuliaFractal();
        ~JuliaFractal() override;

        JuliaFractal(const JuliaFractal& other)             = delete;
        JuliaFractal(JuliaFractal&& other)                  = delete;
        JuliaFractal& operator=(const JuliaFractal& other)  = delete;
        JuliaFractal& operator=(const JuliaFractal&& other) = delete;

    protected:
        virtual void DrawImpl() override;

    private:
        void calculateJuliaOnCPU();
        void calculateJuliaOnGPU();

        int juliaCPU(int x, int y);

        __device__ int juliaGPU(int x, int y);
        friend __global__ void gpuStartCalculation(uint8_t* buffer, JuliaFractal& fractal);

        const size_t ABSOLUTE_VALUE = 1000;
        const size_t IMAGE_WIDTH = 512;
        const size_t IMAGE_HEIGHT = 512;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 4;
        
        std::vector<uint8_t> m_cpuImageBuffer;
        std::vector<uint8_t> m_gpuImageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_cpuTexture = nullptr;
        std::unique_ptr<OpenGL::Resources::Texture> m_gpuTexture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_cpuCalculationTimeText = nullptr;
        std::shared_ptr<UI::Widgets::Texts::Text> m_gpuCalculationTimeText = nullptr;
        
        std::shared_ptr<UI::Widgets::Drags::SingleDrag<float>> m_scaleDrag = nullptr;
        uint64_t m_scaleEventListener = 0;

        std::shared_ptr<UI::Widgets::Inputs::MultipleNumbersInput<float, 2>> m_constantInput = nullptr;
        uint64_t m_constantEventListener = 0;
        
        std::shared_ptr<UI::Widgets::Drags::SingleDrag<uint32_t>> m_iterationsDrag = nullptr;
        uint64_t m_iterationsEventListener = 0;

        std::pair<float, float> m_constant = { -0.8f, 0.156f };
        std::pair<std::pair<float, float>, std::pair<float, float>> m_constantLimit =
        {
            {
                -0.805f,
                -0.795f
            },
            {
                0.125f,
                0.175f
            }
        };

        float m_scale = 1.0f;
        uint32_t m_iterations = 200;
        std::pair<float, float> m_scaleLimit = { 0.1f, 100.0f };
        std::pair<uint32_t, uint32_t> m_iterationsLimit = { 1, 1000 };

        std::atomic<bool> m_isCPUCalculationRunning { false };
        std::atomic<bool> m_isGPUCalculationRunning { false };

        std::thread m_cpuCalculationThread;
        std::thread m_gpuCalculationThread;
    };
}

#endif // CUDA_PANELS_JULIA_FRACTAL_HPP_
