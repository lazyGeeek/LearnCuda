#pragma once
#ifndef CUDA_PANELS_THREADS_SYNC_HPP_
#define CUDA_PANELS_THREADS_SYNC_HPP_

#include "ui/panels/window_panel.hpp"

#include <atomic>
#include <thread>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }
namespace UI::Widgets::Texts { class Text; }

namespace Cuda::Panels
{
    class ThreadsSync : public UI::Panels::WindowPanel
    {
    public:
        ThreadsSync();
        ~ThreadsSync() override;

        ThreadsSync(const ThreadsSync& other)             = delete;
        ThreadsSync(ThreadsSync&& other)                  = delete;
        ThreadsSync& operator=(const ThreadsSync& other)  = delete;
        ThreadsSync& operator=(const ThreadsSync&& other) = delete;

    protected:
        virtual void DrawImpl() override;

    private:
        void calculateAsync();
        void calculateSync();

        friend __global__ void kernel(uint8_t* buffer, ThreadsSync& threadsSync, bool sync);

        const size_t IMAGE_WIDTH = 512;
        const size_t IMAGE_HEIGHT = 512;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 4;
        const size_t THREADS_COUNT = 25;
        
        std::vector<uint8_t> m_asyncImageBuffer;
        std::vector<uint8_t> m_syncImageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_asyncTexture = nullptr;
        std::unique_ptr<OpenGL::Resources::Texture> m_syncTexture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_asyncCalculationTimeText = nullptr;
        std::shared_ptr<UI::Widgets::Texts::Text> m_syncCalculationTimeText = nullptr;

        std::atomic<bool> m_isAsyncCalculationRunning { true };
        std::atomic<bool> m_isSyncCalculationRunning { true };

        std::thread m_asyncCalculationThread;
        std::thread m_syncCalculationThread;
    };
}

#endif // CUDA_PANELS_THREADS_SYNC_HPP_
