#include "cuda/panels/raytracing.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/texts/text.hpp"
#include "utils/time/clock.hpp"

#include <cuda_runtime.h>

#include <iostream>

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

        cudaEvent_t start;
        cudaEvent_t stop;

        cudaError_t error = cudaEventCreate(&start);
        
        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventCreate(&stop);
        
        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventRecord(start, 0);

        uint8_t* buffer = m_noconstImageBuffer.data();
        error = cudaMalloc((void**)&buffer, m_noconstImageBuffer.size() * sizeof(uint8_t));

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        Shapes::Sphere* spheres = nullptr;
        error = cudaMalloc((void**)&spheres, m_spheres.size() * sizeof(Shapes::Sphere));

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaMemcpy(spheres, m_spheres.data(),
                           sizeof(Shapes::Sphere) * SPHERES_COUNT,
                           cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        kernelNonconst<<<blocks, threads>>>(buffer, spheres, *this);

        error = cudaMemcpy(m_noconstImageBuffer.data(), buffer, m_noconstImageBuffer.size(), cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventRecord(stop, 0);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventSynchronize(stop);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        float elapsedTime;

        error = cudaEventElapsedTime( &elapsedTime, start, stop );

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
        
        m_noconstCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", elapsedTime);
 
        error = cudaEventDestroy(start);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
 
        error = cudaEventDestroy(stop);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaFree(spheres);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaFree(buffer);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
    }

    void Raytracing::calculateConst()
    {
        if (!m_isConstCalculationRunning)
            return;

        cudaEvent_t start;
        cudaEvent_t stop;

        cudaError_t error = cudaEventCreate(&start);
        
        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventCreate(&stop);
        
        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventRecord(start, 0);

        uint8_t* buffer = m_constImageBuffer.data();
        error = cudaMalloc((void**)&buffer, m_constImageBuffer.size() * sizeof(uint8_t));

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaMemcpy(m_constSpheres, m_spheres.data(),
                           sizeof(Shapes::Sphere) * SPHERES_COUNT,
                           cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        kernelConst<<<blocks, threads>>>(buffer, *this);

        error = cudaMemcpy(m_constImageBuffer.data(), buffer, m_constImageBuffer.size(), cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
 
        error = cudaEventRecord(stop, 0);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaEventSynchronize(stop);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        float elapsedTime;

        error = cudaEventElapsedTime( &elapsedTime, start, stop );

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
        
        m_constCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", elapsedTime);
 
        error = cudaEventDestroy(start);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
 
        error = cudaEventDestroy(stop);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaFree(buffer);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);
    }
}