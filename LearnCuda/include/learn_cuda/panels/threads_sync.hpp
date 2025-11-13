#pragma once
#ifndef LEARN_CUDA_PANELS_THREADS_SYNC_HPP_
#define LEARN_CUDA_PANELS_THREADS_SYNC_HPP_

#include "ui/panels/window_panel.hpp"

#include <thread>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }
namespace UI::Widgets::Texts { class Text; }

namespace LearnCuda::Panels
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

        friend __global__ void async(uint8_t* buffer, ThreadsSync& threadsSync);
        friend __global__ void sync(uint8_t* buffer, ThreadsSync& threadsSync);

        const size_t IMAGE_WIDTH = 500.0f;
        const size_t IMAGE_HEIGHT = 500.0f;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3;
        const size_t THREADS_COUNT = 25;
        
        std::vector<uint8_t> m_asyncImageBuffer;
        std::vector<uint8_t> m_syncImageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_asyncTexture = nullptr;
        std::unique_ptr<OpenGL::Resources::Texture> m_syncTexture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_asyncCalculationTimeText = nullptr;
        std::shared_ptr<UI::Widgets::Texts::Text> m_syncCalculationTimeText = nullptr;

        bool m_isAsyncCalculationRunning = true;
        bool m_isSyncCalculationRunning = true;

        std::thread m_asyncCalculationThread;
        std::thread m_syncCalculationThread;
    };
}

#endif // LEARN_CUDA_PANELS_THREADS_SYNC_HPP_
