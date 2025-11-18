#include "cuda/panels/raytracing.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/layouts/group.hpp"
#include "ui/widgets/texts/text.hpp"
#include "ui/widgets/visuals/image.hpp"
#include "utils/random.hpp"

#include <cmath>

namespace Cuda::Panels
{
    Raytracing::Raytracing() : UI::Panels::WindowPanel("Raytracing")
    {
        m_noconstImageBuffer.resize(IMAGE_BUFFER_SIZE);
        m_constImageBuffer.resize(IMAGE_BUFFER_SIZE);

        std::fill(m_noconstImageBuffer.begin(), m_noconstImageBuffer.end(), 0);
        std::fill(m_constImageBuffer.begin(), m_constImageBuffer.end(), 0);

        for (int i = 0; i < SPHERES_COUNT; ++i)
        {
            Shapes::Sphere sphere;
            sphere.R = Utils::Random::GetFloat(0.0f, 1.0f);
            sphere.G = Utils::Random::GetFloat(0.0f, 1.0f);
            sphere.B = Utils::Random::GetFloat(0.0f, 1.0f);
            sphere.X = Utils::Random::GetFloat(-250.0f, 250.0f);
            sphere.Y = Utils::Random::GetFloat(-250.0f, 250.0f);
            sphere.Z = Utils::Random::GetFloat(-250.0f, 250.0f);
            sphere.Radius = Utils::Random::GetFloat(20.0f, 100.0f);

            m_spheres.emplace_back(sphere);
        }

        m_noconstTexture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("Non Constant",
                                                                                 m_noconstImageBuffer.data(),
                                                                                 IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                                 OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                                 OpenGL::Settings::ETextureFilteringMode::Linear, false);

        m_constTexture = OpenGL::Resources::Texture::CreateFromMemoryUniquePtr("Constant",
                                                                               m_constImageBuffer.data(),
                                                                               IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                               OpenGL::Settings::ETextureFilteringMode::Linear,
                                                                               OpenGL::Settings::ETextureFilteringMode::Linear, false);


        UI::Settings::GroupWidgetSettings groupSettings;
        groupSettings.FrameStyle = false;
        groupSettings.AutoResizeY = true;
        groupSettings.AutoResizeX = true;

        std::shared_ptr<UI::Widgets::Layouts::Group> asyncImageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        asyncImageGroup->SetSameLine(true);
        asyncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Noconst Calculation");
        m_noconstCalculationTimeText = asyncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 milliseconds");

        OpenGL::Resources::TextureInfo info = m_noconstTexture->GetInfo();
        asyncImageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        std::shared_ptr<UI::Widgets::Layouts::Group> syncImageGroup = CreateWidget<UI::Widgets::Layouts::Group>(groupSettings);
        syncImageGroup->SetSameLine(true);
        syncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Const Calculation");
        m_constCalculationTimeText = syncImageGroup->CreateWidget<UI::Widgets::Texts::Text>("Calculation time: 0.0 milliseconds");

        info = m_constTexture->GetInfo();
        syncImageGroup->CreateWidget<UI::Widgets::Visuals::Image>(info.Id, info.Width, info.Height);

        OpenEvent += [&]()
        {
            m_noconstCalculationThread = std::thread([&]()
            {
                m_isNoconstCalculationRunning = true;

                while (m_isNoconstCalculationRunning)
                {
                    if (IsOpened()) calculateNoconst();
                }
            });
            
            m_constCalculationThread = std::thread([&]()
            {
                m_isConstCalculationRunning = true;

                while (m_isConstCalculationRunning)
                {
                    if (IsOpened()) calculateConst();
                }
            });
        };

        CloseEvent += [&]()
        {
            m_isNoconstCalculationRunning = false;
            m_isConstCalculationRunning = false;
            
            if (m_noconstCalculationThread.joinable())
                m_noconstCalculationThread.join();

            if (m_constCalculationThread.joinable())
                m_constCalculationThread.join();
        };
    }

    Raytracing::~Raytracing()
    {
        m_isNoconstCalculationRunning = false;
        m_isConstCalculationRunning = false;

        if (m_noconstCalculationThread.joinable())
            m_noconstCalculationThread.join();

        if (m_constCalculationThread.joinable())
            m_constCalculationThread.join();

        RemoveAllWidgets();

        if (m_noconstTexture)
            m_noconstTexture = nullptr;

        if (m_constTexture)
            m_constTexture = nullptr;
    }

    void Raytracing::DrawImpl()
    {
        m_noconstTexture->UpdateTexture(m_noconstImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);
        m_constTexture->UpdateTexture(m_constImageBuffer.data(), IMAGE_WIDTH, IMAGE_HEIGHT);

        UI::Panels::WindowPanel::DrawImpl();
    }
}
