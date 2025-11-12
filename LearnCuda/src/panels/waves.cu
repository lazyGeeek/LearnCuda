#include "learn_cuda/panels/waves.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/texts/text.hpp"
#include "utils/time/clock.hpp"

#include <cuda_runtime.h>

namespace LearnCuda::Panels
{
    __device__ int Waves::wavesGPU(int x, int y, float ticks)
    {
        float fx = x - IMAGE_WIDTH / 2.0f;
        float fy = y - IMAGE_HEIGHT / 2.0f;
        float d = sqrt(fx * fx + fy * fy);

        return static_cast<int>(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));
    }

    __global__ void gpuStartCalculation(uint8_t* buffer, float ticks, Waves& waves)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x; 
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = 3 * x + y * 3 * blockDim.x * gridDim.x;

        int color = waves.wavesGPU(x, y, ticks);

        buffer[offset + 0] = color;
        buffer[offset + 1] = color;
        buffer[offset + 2] = color;
    }

    void Waves::calculateWavesOnGPU(int64_t ticks)
    {
        if (!m_isGPUCalculationRunning)
            return;

        Utils::Time::Clock clock;
        clock.Start();

        uint8_t* buffer = m_gpuImageBuffer.data();
        cudaError_t error = cudaMalloc((void**)&buffer, m_gpuImageBuffer.size() * sizeof(uint8_t));

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        dim3 blocks(IMAGE_WIDTH / 25, IMAGE_HEIGHT / 25);
        dim3 threads(25, 25);

        gpuStartCalculation<<<blocks, threads>>>(buffer, ticks / 50.0f, *this);

        error = cudaMemcpy(m_gpuImageBuffer.data(), buffer, m_gpuImageBuffer.size(), cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaFree(buffer);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        m_gpuCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", clock.GetMicroseconds() / 1000.0f);
    }
}