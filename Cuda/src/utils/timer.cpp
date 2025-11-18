#include "cuda/utils/timer.hpp"
#include "cuda/utils/cuda_helper.hpp"

namespace Cuda::Utils
{
    Timer::~Timer()
    {
        if (m_isRunning) Stop();
    }

    void Timer::Start()
    {
        CUDA_HANDLE_ERROR(cudaEventCreate(&m_start));
        CUDA_HANDLE_ERROR(cudaEventCreate(&m_stop));
        CUDA_HANDLE_ERROR(cudaEventRecord(m_start, 0));

        m_isRunning = true;
    }

    float Timer::Stop()
    {
        m_isRunning = false;

        CUDA_HANDLE_ERROR(cudaEventRecord(m_stop, 0));
        CUDA_HANDLE_ERROR(cudaEventSynchronize(m_stop));

        float elapsedTime = 0.0f;

        CUDA_HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, m_start, m_stop)); 
        CUDA_HANDLE_ERROR(cudaEventDestroy(m_start));
        CUDA_HANDLE_ERROR(cudaEventDestroy(m_stop));
        
        return elapsedTime;
    }
}
