#include "cuda/panels/ripples.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/layouts/group.hpp"
#include "ui/widgets/texts/text.hpp"
#include "ui/widgets/visuals/image.hpp"
#include "utils/time/clock.hpp"

#include <cmath>

namespace Cuda::Panels
{
    Ripples::Ripples() : UI::Panels::WindowPanel("Ripples")
    {
        m_cpuImageBuffer.resize(IMAGE_BUFFER_SIZE);
        m_gpuImageBuffer.resize(IMAGE_BUFFER_SIZE);

        std::fill(m_cpuImageBuffer.begin(), m_cpuImageBuffer.end(), 0);
        std::fill(m_gpuImageBuffer.begin(), m_gpuImageBuffer.end(), 0);

        m_cpuTexture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("CPU Calculation",
                                                                             m_cpuImageBuffer.data(),
                                                                             IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                             OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                             OpenGL::Settings::ETextureFilteringMode::Linear, false);

        m_gpuTexture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("GPU Calculation",
                                                                             m_gpuImageBuffer.data(),
                                                                             IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                             OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                             OpenGL::Settings::ETextureFilteringMode::Linear, false);


        UI::Settings::GroupWidgetSettings groupSettings;
        groupSettings.FrameStyle = false;
        groupSettings.AutoResizeY = true;
        groupSettings.AutoResizeX = true;
        
        std::shared_ptr<UI::Widgets::Layouts::Group> cpuImageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        cpuImageGroup->SetSameLine(true);
        cpuImageGroup->CreateWidget<UI::Widgets::Texts::Text>("CPU Calculation");
        m_cpuCalculationTimeText = cpuImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 ms");

        OpenGL::Resources::TextureInfo info = m_cpuTexture->GetInfo();
        cpuImageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        std::shared_ptr<UI::Widgets::Layouts::Group> gpuImageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        gpuImageGroup->SetSameLine(true);
        gpuImageGroup->CreateWidget<UI::Widgets::Texts::Text>("GPU Calculation");
        m_gpuCalculationTimeText = gpuImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 ms");

        info = m_gpuTexture->GetInfo();
        gpuImageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        OpenEvent += [&]()
        {
            m_cpuCalculationThread = std::thread([&]()
            {
                m_isCPUCalculationRunning = true;
                Utils::Time::Clock ticks;
                ticks.Start();

                while (m_isCPUCalculationRunning)
                {
                    if (IsOpened()) calculateRipplesOnCPU(ticks.GetMilliseconds());
                }
            });

            m_gpuCalculationThread = std::thread([&]()
            {
                m_isGPUCalculationRunning = true;
                Utils::Time::Clock ticks;
                ticks.Start();

                while (m_isGPUCalculationRunning)
                {
                    if (IsOpened()) calculateRipplesOnGPU(ticks.GetMilliseconds());
                }
            });
        };

        CloseEvent += [&]()
        {
            m_isCPUCalculationRunning = false;
            m_isGPUCalculationRunning = false;
            
            if (m_cpuCalculationThread.joinable())
                m_cpuCalculationThread.join();

            if (m_gpuCalculationThread.joinable())
                m_gpuCalculationThread.join();
        };
    }

    Ripples::~Ripples()
    {
        m_isCPUCalculationRunning = false;
        m_isGPUCalculationRunning = false;

        if (m_cpuCalculationThread.joinable())
            m_cpuCalculationThread.join();

        if (m_gpuCalculationThread.joinable())
            m_gpuCalculationThread.join();

        RemoveAllWidgets();

        if (m_cpuTexture)
            m_cpuTexture = nullptr;

        if (m_gpuTexture)
            m_gpuTexture = nullptr;
    }

    void Ripples::DrawImpl()
    {
        m_cpuTexture->UpdateTexture(m_cpuImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
        m_gpuTexture->UpdateTexture(m_gpuImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);

        UI::Panels::WindowPanel::DrawImpl();
    }

    int Ripples::ripplesCPU(int x, int y, float ticks)
    {
        float fx = x - IMAGE_WIDTH / 2.0f;
        float fy = y - IMAGE_HEIGHT / 2.0f;
        float d = sqrt(fx * fx + fy * fy);

        return static_cast<int>(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
    }

    void Ripples::calculateRipplesOnCPU(int64_t ticks)
    {
        if (!m_isCPUCalculationRunning)
            return;

        Utils::Time::Clock clock;
        clock.Start();

        for (int y = 0; y < IMAGE_HEIGHT; ++y)
        {
            for (int x = 0; x < IMAGE_WIDTH; ++x)
            {
                int offset = x * 3 + y * 3 * IMAGE_WIDTH;

                int color = ripplesCPU(x, y, ticks / 50.0f);

                m_cpuImageBuffer[offset + 0] = color;
                m_cpuImageBuffer[offset + 1] = color;
                m_cpuImageBuffer[offset + 2] = color;
            }
        }

        m_cpuCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", clock.GetMicroseconds() / 1000.0f);
    }
}
