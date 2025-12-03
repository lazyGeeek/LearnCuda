#include "cuda/panels/heat.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/layouts/group.hpp"
#include "ui/widgets/texts/text.hpp"
#include "ui/widgets/visuals/image.hpp"
#include "utils/time/clock.hpp"
#include "utils/random.hpp"

#include <iostream>

namespace Cuda::Panels
{
    Heat::Heat() : UI::Panels::WindowPanel("Heat")
    {
        m_imageBuffer.resize(IMAGE_BUFFER_SIZE);

        m_texture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("Non Constant",
                                                                            reinterpret_cast<uint8_t*>(m_imageBuffer.data()),
                                                                            IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                            OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                            OpenGL::Settings::ETextureFilteringMode::Linear, false);

        UI::Settings::GroupWidgetSettings groupSettings;
        groupSettings.FrameStyle = false;
        groupSettings.AutoResizeY = true;
        groupSettings.AutoResizeX = true;

        std::shared_ptr<UI::Widgets::Layouts::Group> imageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        imageGroup->SetSameLine(true);
        imageGroup->CreateWidget<UI::Widgets::Texts::Text>("1D Calculation");
        m_calculationTimeText = imageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 milliseconds");

        OpenGL::Resources::TextureInfo info = m_texture->GetInfo();
        imageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        OpenEvent += [&]()
        {
            m_calculationThread = std::thread([&]()
            {
                m_isCalculationRunning.store(true, std::memory_order_release);

                Utils::Time::Clock clock;
                clock.Start();

                std::fill(m_imageBuffer.begin(), m_imageBuffer.end(), make_uchar4(0, 0, 0, 255));

                int middleWidth = IMAGE_WIDTH / 2;
                int middleHeight = IMAGE_HEIGHT / 2;

                for (int y = IMAGE_HEIGHT / 2 - 100; y < IMAGE_HEIGHT / 2 + 100; ++y)
                {
                    for (int x = IMAGE_WIDTH / 2 - 100; x < IMAGE_WIDTH / 2 + 100; ++x)
                    {
                        m_imageBuffer[x + y * IMAGE_WIDTH] = make_uchar4(255, 255, 255, 255);
                    }
                }

                for (int i = 0; i < 500; ++i)
                {
                    if (IsOpened()) calculate();
                }

                m_calculationTimeText->SetContent("Calculation time: %.3f milliseconds", clock.GetMicroseconds() / 1000.0f);
                m_isCalculationRunning.store(false, std::memory_order_release);
            });
        };

        CloseEvent += [&]()
        {
            m_isCalculationRunning.store(false, std::memory_order_release);
            
            if (m_calculationThread.joinable())
                m_calculationThread.join();
        };
    }

    Heat::~Heat()
    {
        m_isCalculationRunning.store(false, std::memory_order_release);

        if (m_calculationThread.joinable())
            m_calculationThread.join();

        RemoveAllWidgets();

        if (m_texture)
            m_texture = nullptr;
    }

    void Heat::DrawImpl()
    {
        m_texture->UpdateTexture(reinterpret_cast<uint8_t*>(m_imageBuffer.data()), IMAGE_WIDTH, IMAGE_HEIGHT);

        UI::Panels::WindowPanel::DrawImpl();
    }
}
