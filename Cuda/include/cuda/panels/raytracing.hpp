#pragma once
#ifndef CUDA_PANELS_RAYTRACING_HPP_
#define CUDA_PANELS_RAYTRACING_HPP_

#include "ui/panels/window_panel.hpp"
#include "cuda/shapes/sphere.hpp"

#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace OpenGL::Resources { class Texture; }
namespace UI::Widgets::Texts { class Text; }

namespace Cuda::Panels
{
    class Raytracing : public UI::Panels::WindowPanel
    {
    public:
        Raytracing();
        ~Raytracing() override;

        Raytracing(const Raytracing& other)             = delete;
        Raytracing(Raytracing&& other)                  = delete;
        Raytracing& operator=(const Raytracing& other)  = delete;
        Raytracing& operator=(const Raytracing&& other) = delete;

    protected:
        virtual void DrawImpl() override;

    private:
        void calculateNoconst();
        void calculateConst();

        friend __global__ void kernelNonconst(uint8_t* buffer, Shapes::Sphere* spheres, Raytracing& raytracing);
        friend __global__ void kernelConst(uint8_t* buffer, Raytracing& raytracing);

        const size_t IMAGE_WIDTH = 500.0f;
        const size_t IMAGE_HEIGHT = 500.0f;
        const size_t IMAGE_BUFFER_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3;
        const size_t THREADS_COUNT = 25;
        const size_t SPHERES_COUNT = 10;
        
        __constant__ Shapes::Sphere m_constSpheres[ 10 ];

        std::vector<Shapes::Sphere> m_spheres;

        std::vector<uint8_t> m_noconstImageBuffer;
        std::vector<uint8_t> m_constImageBuffer;

        std::unique_ptr<OpenGL::Resources::Texture> m_noconstTexture = nullptr;
        std::unique_ptr<OpenGL::Resources::Texture> m_constTexture = nullptr;

        std::shared_ptr<UI::Widgets::Texts::Text> m_noconstCalculationTimeText = nullptr;
        std::shared_ptr<UI::Widgets::Texts::Text> m_constCalculationTimeText = nullptr;

        bool m_isNoconstCalculationRunning = true;
        bool m_isConstCalculationRunning = true;

        std::thread m_noconstCalculationThread;
        std::thread m_constCalculationThread;
    };
}

#endif // CUDA_PANELS_RAYTRACING_HPP_
