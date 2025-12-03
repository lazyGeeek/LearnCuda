
#pragma once
#ifndef CUDA_PANELS_HEAT_HPP_
#define CUDA_PANELS_HEAT_HPP_

#include "ui/panels/window_panel.hpp"

#include <atomic>
#include <cmath>
#include <thread>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }
namespace UI::Widgets::Texts { class Text; }

namespace Cuda::Panels
{
    class Heat : public UI::Panels::WindowPanel
    {
    public:
        Heat();
        ~Heat() override;

        Heat(const Heat& other)             = delete;
        Heat(Heat&& other)                  = delete;
        Heat& operator=(const Heat& other)  = delete;
        Heat& operator=(const Heat&& other) = delete;

    protected:
        virtual void DrawImpl() override;

    private:
        void calculate();

        friend __global__ void kernel(uchar4* output, cudaTextureObject_t texObj, int width, int height, Heat& heat);

        const size_t IMAGE_WIDTH = 512;
        const size_t IMAGE_HEIGHT = 512;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 4;
        const size_t THREADS_COUNT = 16;
        const float SPEED = 0.25f;

        dim3 blocks = dim3(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads = dim3(THREADS_COUNT, THREADS_COUNT);

        std::vector<uchar4> m_imageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_texture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_calculationTimeText = nullptr;

        std::atomic<bool> m_isCalculationRunning { true };

        std::thread m_calculationThread;
    };
}

#endif // CUDA_PANELS_HEAT_HPP_