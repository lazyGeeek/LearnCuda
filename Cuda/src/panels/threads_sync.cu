#include "cuda/panels/threads_sync.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "cuda/utils/timer.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/texts/text.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace Cuda::Panels
{
    #define PI 3.1415926535897932f

    __global__ void kernel(uint8_t* buffer, ThreadsSync& threadsSync, bool sync)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x; 
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = x + y * blockDim.x * gridDim.x;

        __shared__ float shared[25][25];

        const float period = 128.0f;

        shared[threadIdx.x][threadIdx.y] =
                255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
                    (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

        if (sync)
            __syncthreads();

        buffer[offset * 4 + 0] = 0;
        buffer[offset * 4 + 1] = shared[25 - 1 - threadIdx.x][25 - 1 - threadIdx.y];
        buffer[offset * 4 + 2] = 0;
        buffer[offset * 4 + 3] = 255;
    }

    void ThreadsSync::calculateAsync()
    {
        if (!m_isAsyncCalculationRunning.load(std::memory_order_acquire))
            return;

        Utils::Timer timer;
        timer.Start();

        uint8_t* buffer = m_asyncImageBuffer.data();
        CUDA_HANDLE_ERROR(cudaMalloc((void**)&buffer, m_asyncImageBuffer.size() * sizeof(uint8_t)));

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        kernel<<<blocks, threads>>>(buffer, *this, false);

        CUDA_HANDLE_ERROR(cudaMemcpy(m_asyncImageBuffer.data(), buffer, m_asyncImageBuffer.size(), cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaFree(buffer));
        
        m_asyncCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", timer.Stop());
    }

    void ThreadsSync::calculateSync()
    {
        if (!m_isSyncCalculationRunning.load(std::memory_order_acquire))
            return;

        Utils::Timer timer;
        timer.Start();

        uint8_t* buffer = m_syncImageBuffer.data();
        CUDA_HANDLE_ERROR(cudaMalloc((void**)&buffer, m_syncImageBuffer.size() * sizeof(uint8_t)));

        dim3 blocks(IMAGE_WIDTH / THREADS_COUNT, IMAGE_HEIGHT / THREADS_COUNT);
        dim3 threads(THREADS_COUNT, THREADS_COUNT);

        kernel<<<blocks, threads>>>(buffer, *this, true);

        CUDA_HANDLE_ERROR(cudaMemcpy(m_syncImageBuffer.data(), buffer, m_syncImageBuffer.size(), cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaFree(buffer));

        m_syncCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", timer.Stop());
    }
}