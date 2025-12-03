#include "cuda/panels/ripples.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "cuda/utils/timer.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/texts/text.hpp"

#include <cuda_runtime.h>

namespace Cuda::Panels
{
    __global__ void ripplesGpu(uint8_t* buffer, float ticks, Ripples& ripples)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x; 
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = x + y * blockDim.x * gridDim.x;

        float fx = x - ripples.IMAGE_WIDTH / 2.0f;
        float fy = y - ripples.IMAGE_HEIGHT / 2.0f;
        float d = sqrt(fx * fx + fy * fy);

        int color = static_cast<int>(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

        buffer[offset * 4 + 0] = color;
        buffer[offset * 4 + 1] = color;
        buffer[offset * 4 + 2] = color;
        buffer[offset * 4 + 3] = color;
    }

    void Ripples::calculateRipplesOnGPU(int64_t ticks)
    {
        if (!m_isGPUCalculationRunning.load(std::memory_order_acquire))
            return;

        Utils::Timer timer;
        timer.Start();

        uint8_t* buffer = m_gpuImageBuffer.data();
        CUDA_HANDLE_ERROR(cudaMalloc((void**)&buffer, m_gpuImageBuffer.size() * sizeof(uint8_t)));

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        ripplesGpu<<<blocks, threads>>>(buffer, ticks / 50.0f, *this);

        CUDA_HANDLE_ERROR(cudaMemcpy(m_gpuImageBuffer.data(), buffer, m_gpuImageBuffer.size(), cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaFree(buffer));

        m_gpuCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", timer.Stop());
    }
}