#include "cuda/panels/raytracing.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "cuda/utils/timer.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/texts/text.hpp"

#include <cuda_runtime.h>

namespace Cuda::Panels
{
    #define PI 3.1415926535897932f

    __global__ void kernelNonconst(uint8_t* buffer, Shapes::Sphere* spheres, Raytracing& raytracing)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x; 
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = 3 * x + y * 3 * blockDim.x * gridDim.x;

        float midX = raytracing.IMAGE_WIDTH / 2.0f;
        float midY = raytracing.IMAGE_HEIGHT / 2.0f;
        float ox = x - midX;
        float oy = y - midY;

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float maxz = FLT_MIN;

        for (int i = 0; i < raytracing.SPHERES_COUNT; ++i)
        {
            Shapes::Sphere& sphere = spheres[i];
            float fscale = 0.0f;
            float t = sphere.Hit(ox, oy, fscale);
            if (t > maxz)
            {
                r = sphere.R * fscale;
                g = sphere.G * fscale;
                b = sphere.B * fscale;
                maxz = t;
            }
        }

        buffer[offset + 0] = static_cast<int>(r * 255);
        buffer[offset + 1] = static_cast<int>(g * 255);
        buffer[offset + 2] = static_cast<int>(b * 255);
    }

    __global__ void kernelConst(uint8_t* buffer, Raytracing& raytracing)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x; 
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = 3 * x + y * 3 * blockDim.x * gridDim.x;

        float midX = raytracing.IMAGE_WIDTH / 2.0f;
        float midY = raytracing.IMAGE_HEIGHT / 2.0f;
        float ox = x - midX;
        float oy = y - midY;

        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        float maxz = FLT_MIN;

        for (int i = 0; i < raytracing.SPHERES_COUNT; ++i)
        {
            Shapes::Sphere& sphere = raytracing.m_constSpheres[i];
            float fscale = 0.0f;
            float t = sphere.Hit(ox, oy, fscale);
            if (t > maxz)
            {
                r = sphere.R * fscale;
                g = sphere.G * fscale;
                b = sphere.B * fscale;
                maxz = t;
            }
        }

        buffer[offset + 0] = static_cast<int>(r * 255);
        buffer[offset + 1] = static_cast<int>(g * 255);
        buffer[offset + 2] = static_cast<int>(b * 255);
    }

    void Raytracing::calculateNoconst()
    {
        if (!m_isNoconstCalculationRunning)
            return;

        Utils::Timer timer;
        timer.Start();
        
        uint8_t* buffer = m_noconstImageBuffer.data();
        CUDA_HANDLE_ERROR(cudaMalloc((void**)&buffer, m_noconstImageBuffer.size() * sizeof(uint8_t)));

        Shapes::Sphere* spheres = nullptr;
        CUDA_HANDLE_ERROR(cudaMalloc((void**)&spheres, m_spheres.size() * sizeof(Shapes::Sphere)));
        CUDA_HANDLE_ERROR(cudaMemcpy(spheres, m_spheres.data(),
                          sizeof(Shapes::Sphere) * SPHERES_COUNT,
                          cudaMemcpyHostToDevice));

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        kernelNonconst<<<blocks, threads>>>(buffer, spheres, *this);

        CUDA_HANDLE_ERROR(cudaMemcpy(m_noconstImageBuffer.data(), buffer, m_noconstImageBuffer.size(), cudaMemcpyDeviceToHost));        
        CUDA_HANDLE_ERROR(cudaFree(spheres));
        CUDA_HANDLE_ERROR(cudaFree(buffer));

        m_noconstCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", timer.Stop());
    }

    void Raytracing::calculateConst()
    {
        if (!m_isConstCalculationRunning)
            return;

        Utils::Timer timer;
        timer.Start();

        uint8_t* buffer = m_constImageBuffer.data();

        CUDA_HANDLE_ERROR(cudaMalloc((void**)&buffer, m_constImageBuffer.size() * sizeof(uint8_t)));
        CUDA_HANDLE_ERROR(cudaMemcpy(m_constSpheres, m_spheres.data(),
                          sizeof(Shapes::Sphere) * SPHERES_COUNT,
                          cudaMemcpyHostToDevice));

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        kernelConst<<<blocks, threads>>>(buffer, *this);

        CUDA_HANDLE_ERROR(cudaMemcpy(m_constImageBuffer.data(), buffer, m_constImageBuffer.size(), cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaFree(buffer));

        m_constCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", timer.Stop());
    }
}