#ifndef CUDA_UTILS_TIMER_HPP_
#define CUDA_UTILS_TIMER_HPP_

#include <cuda_runtime.h>

namespace Cuda::Utils
{
    class Timer
    {
    public:
        Timer() = default;
        ~Timer();

        Timer(const Timer& other)             = delete;
        Timer(Timer&& other)                  = delete;
        Timer& operator=(const Timer& other)  = delete;
        Timer& operator=(const Timer&& other) = delete;

        void Start();
        float Stop();
    
    private:
        cudaEvent_t m_start;
        cudaEvent_t m_stop;
        
        bool m_isRunning = false;
    };
}

#endif // CUDA_UTILS_TIMER_HPP_