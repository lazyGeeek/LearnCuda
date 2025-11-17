#include "cuda/panels/threads_sync.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/layouts/group.hpp"
#include "ui/widgets/texts/text.hpp"
#include "ui/widgets/visuals/image.hpp"

#include <cmath>
#include <iostream>

namespace Cuda::Panels
{
    ThreadsSync::ThreadsSync() : UI::Panels::WindowPanel("Thread Sync")
    {
        m_asyncImageBuffer.resize(IMAGE_BUFFER_SIZE);
        m_syncImageBuffer.resize(IMAGE_BUFFER_SIZE);

        std::fill(m_asyncImageBuffer.begin(), m_asyncImageBuffer.end(), 0);
        std::fill(m_syncImageBuffer.begin(), m_syncImageBuffer.end(), 0);

        m_asyncTexture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("Async Calculation",
                                                                               m_asyncImageBuffer.data(),
                                                                               IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                               OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                               OpenGL::Settings::ETextureFilteringMode::Linear, false);

        m_syncTexture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("Sync Calculation",
                                                                              m_syncImageBuffer.data(),
                                                                              IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                              OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                              OpenGL::Settings::ETextureFilteringMode::Linear, false);


        UI::Settings::GroupWidgetSettings groupSettings;
        groupSettings.FrameStyle = false;
        groupSettings.AutoResizeY = true;
        groupSettings.AutoResizeX = true;

        std::shared_ptr<UI::Widgets::Layouts::Group> asyncImageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        asyncImageGroup->SetSameLine(true);
        asyncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Async Calculation");
        m_asyncCalculationTimeText = asyncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 milliseconds");

        OpenGL::Resources::TextureInfo info = m_asyncTexture->GetInfo();
        asyncImageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        std::shared_ptr<UI::Widgets::Layouts::Group> syncImageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        syncImageGroup->SetSameLine(true);
        syncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Sync Calculation");
        m_syncCalculationTimeText = syncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 milliseconds");

        info = m_syncTexture->GetInfo();
        syncImageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        OpenEvent += [&]()
        {
            m_asyncCalculationThread = std::thread([&]()
            {
                m_isAsyncCalculationRunning = true;

                while (m_isAsyncCalculationRunning)
                {
                    if (IsOpened()) calculateAsync();
                }
            });
            
            m_syncCalculationThread = std::thread([&]()
            {
                m_isSyncCalculationRunning = true;

                while (m_isSyncCalculationRunning)
                {
                    if (IsOpened()) calculateSync();
                }
            });
        };

        CloseEvent += [&]()
        {
            m_isAsyncCalculationRunning = false;
            m_isSyncCalculationRunning = false;
            
            if (m_asyncCalculationThread.joinable())
                m_asyncCalculationThread.join();

            if (m_syncCalculationThread.joinable())
                m_syncCalculationThread.join();
        };
    }

    ThreadsSync::~ThreadsSync()
    {
        m_isAsyncCalculationRunning = false;
        m_isSyncCalculationRunning = false;

        if (m_asyncCalculationThread.joinable())
            m_asyncCalculationThread.join();

        if (m_syncCalculationThread.joinable())
            m_syncCalculationThread.join();

        RemoveAllWidgets();

        if (m_asyncTexture)
            m_asyncTexture = nullptr;

        if (m_syncTexture)
            m_syncTexture = nullptr;
    }

    void ThreadsSync::DrawImpl()
    {
        m_asyncTexture->UpdateTexture(m_asyncImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
        m_syncTexture->UpdateTexture(m_syncImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);

        UI::Panels::WindowPanel::DrawImpl();
    }
}
