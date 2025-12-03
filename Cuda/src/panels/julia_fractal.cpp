#include "cuda/panels/julia_fractal.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/drags/single_drag.hpp"
#include "ui/widgets/inputs/multiple_numbers_input.hpp"
#include "ui/widgets/inputs/single_number_input.hpp"
#include "ui/widgets/layouts/group.hpp"
#include "ui/widgets/texts/text.hpp"
#include "ui/widgets/visuals/image.hpp"
#include "utils/time/clock.hpp"

namespace Cuda::Panels
{
    JuliaFractal::JuliaFractal() : UI::Panels::WindowPanel("Julia Fractal")
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
        
        std::shared_ptr<UI::Widgets::Layouts::Group> configGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        m_scaleDrag = configGroup->CreateWidget<UI::Widgets::Drags::SingleDrag<float>>("Scale",
                                                                                       m_scaleLimit.first,
                                                                                       m_scaleLimit.second,
                                                                                       m_scale, 0.1f);
        m_scaleDrag->SetSameLine(true);
        m_scaleEventListener = m_scaleDrag->ValueChangedEvent += ([&](float value)
        {
            m_scale = value;
        });
        
        std::array<float, 2> constants = { m_constant.first, m_constant.second };
        m_constantInput = configGroup->CreateWidget<UI::Widgets::Inputs::MultipleNumbersInput<float, 2>>("Constant",
                                                                                                         constants,
                                                                                                         0.001f,
                                                                                                         0.001f);
        m_constantInput->SetSameLine(true);
        m_constantEventListener = m_constantInput->ContentChangedEvent += ([&](std::array<float, 2> values)
        {

            m_constant = { values[0], values[1] };
        });

        m_iterationsDrag = configGroup->CreateWidget<UI::Widgets::Drags::SingleDrag<uint32_t>>("Iterations",
                                                                                               m_iterationsLimit.first,
                                                                                               m_iterationsLimit.second,
                                                                                               m_iterations, 1);
        m_iterationsDrag->SetSameLine(true);
        m_iterationsEventListener = m_iterationsDrag->ValueChangedEvent += ([&](uint32_t content)
        {
            if (content <= m_iterationsLimit.first)
            {
                content = m_iterationsLimit.first;
                m_iterationsDrag->SetValue(content);
            }
            else if (content >= m_iterationsLimit.second)
            {
                content = m_iterationsLimit.second;
                m_iterationsDrag->SetValue(content);
            }
            m_iterations = content;
        });

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
                m_isCPUCalculationRunning.store(true, std::memory_order_release);

                while (m_isCPUCalculationRunning.load(std::memory_order_acquire))
                {
                    if (IsOpened()) calculateJuliaOnCPU();
                }
            });

            m_gpuCalculationThread = std::thread([&]()
            {
                m_isGPUCalculationRunning.store(true, std::memory_order_release);

                while (m_isGPUCalculationRunning.load(std::memory_order_acquire))
                {
                    if (IsOpened()) calculateJuliaOnGPU();
                }
            });
        };

        CloseEvent += [&]()
        {
            m_isCPUCalculationRunning.store(false, std::memory_order_release);
            m_isGPUCalculationRunning.store(false, std::memory_order_release);
            
            if (m_cpuCalculationThread.joinable())
                m_cpuCalculationThread.join();

            if (m_gpuCalculationThread.joinable())
                m_gpuCalculationThread.join();
        };
    }

    JuliaFractal::~JuliaFractal()
    {
        m_isCPUCalculationRunning.store(false, std::memory_order_release);
        m_isGPUCalculationRunning.store(false, std::memory_order_release);

        if (m_cpuCalculationThread.joinable())
            m_cpuCalculationThread.join();

        if (m_gpuCalculationThread.joinable())
            m_gpuCalculationThread.join();

        if (m_scaleDrag)
            m_scaleDrag->ValueChangedEvent -= m_scaleEventListener;

        if (m_constantInput)
            m_constantInput->ContentChangedEvent -= m_constantEventListener;

        if (m_iterationsDrag)
            m_iterationsDrag->ValueChangedEvent -= m_iterationsEventListener;

        RemoveAllWidgets();

        if (m_cpuTexture)
            m_cpuTexture = nullptr;

        if (m_gpuTexture)
            m_gpuTexture = nullptr;
    }

    void JuliaFractal::DrawImpl()
    {
        m_cpuTexture->UpdateTexture(m_cpuImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
        m_gpuTexture->UpdateTexture(m_gpuImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);

        UI::Panels::WindowPanel::DrawImpl();
    }

    int JuliaFractal::juliaCPU(int x, int y)
    {
        float jx = ((IMAGE_WIDTH / 2.0f - x) / (IMAGE_WIDTH / 2.0f)) / m_scale;
        float jy = ((IMAGE_HEIGHT / 2.0f - y) / (IMAGE_HEIGHT / 2.0f)) / m_scale;

        std::pair<float, float> curr(jx, jy);

        for (int i = 0; i < m_iterations; i++)
        {
            std::pair<float, float> temp(0.0f, 0.0f);
            temp.first = (curr.first * curr.first) - (curr.second * curr.second);
            temp.second = (curr.second * curr.first) + (curr.first * curr.second);
            curr = temp;

            curr.first = curr.first + m_constant.first;
            curr.second = curr.second + m_constant.second;

            float magnitude = curr.first * curr.first + curr.second * curr.second;
            if (magnitude > ABSOLUTE_VALUE)
                return 0;
        }

        return 1;
    }

    void JuliaFractal::calculateJuliaOnCPU()
    {
        if (!m_isCPUCalculationRunning)
            return;

        Utils::Time::Clock clock;
        clock.Start();

        for (int y = 0; y < IMAGE_HEIGHT; ++y)
        {
            for (int x = 0; x < IMAGE_WIDTH; ++x)
            {
                int offset = x + y * IMAGE_WIDTH;

                int color = 255 * juliaCPU(x, y);

                m_cpuImageBuffer[offset * 4 + 0] = color;
                m_cpuImageBuffer[offset * 4 + 1] = color;
                m_cpuImageBuffer[offset * 4 + 2] = color;
                m_cpuImageBuffer[offset * 4 + 3] = color;
            }
        }

        m_cpuCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", clock.GetMicroseconds() / 1000.0f);
    }
}
