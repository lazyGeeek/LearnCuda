#include "cuda/device.hpp"
#include "cuda/utils/cuda_helper.hpp"

namespace Cuda
{
    cudaDeviceProp Device::GetProperties(int index)
    {
        cudaDeviceProp properties;
        cudaError_t error = cudaSetDevice(index);

        if (cudaSuccess != error)
        {
            Utils::CudaHelper::PrintCudaError(error);
            return properties;
        }
   
        error = cudaGetDeviceProperties(&properties, index);

        if (cudaSuccess != error)
            Utils::CudaHelper::PrintCudaError(error);

        return properties;
    }

    int Device::GetClockRate(int index)
    {
        int clockRate = -1;

        cudaError_t error = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, index);

        if (cudaSuccess != error)
            Utils::CudaHelper::PrintCudaError(error);

        return clockRate;
    }

    int Device::GetMemoryClockRate(int index)
    {
        int memoryClockRate = -1;
#if CUDART_VERSION >= 5000
#if CUDART_VERSION >= 13000
        cudaError_t error = cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, index);
        if (cudaSuccess != error)
        {
            Utils::CudaHelper::PrintCudaError(error);
            return memoryClockRate;
        }
#else
        cudaDeviceProp properties;
        cudaError_t error = cudaGetDeviceProperties(&properties, index);
        if (cudaSuccess != error)
        {
            Utils::CudaHelper::PrintCudaError(error);
            return memoryClockRate;
        }
        memoryClockRate = properties.memoryClockRate;
#endif
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the
        // CUDA Driver API)
        getCudaAttribute<int>(&memoryClockRate, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, index);
#endif
        return memoryClockRate * 1e-3f;
    }

    int Device::GetComputeMode(int index)
    {
        int computeMode = -1;

        cudaError_t error = cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, index);

        if (cudaSuccess != error)
            Utils::CudaHelper::PrintCudaError(error);

        return computeMode;
    }

    bool Device::IsAttributeGpuOverlap(int index)
    {
        int gpuOverlap = -1;

        cudaError_t error = cudaDeviceGetAttribute(&gpuOverlap, cudaDevAttrGpuOverlap, index);

        if (cudaSuccess != error)
            Utils::CudaHelper::PrintCudaError(error);

        return gpuOverlap;
    }

    bool Device::IsKernelExecTimeout(int index)
    {
        int kernelExecTimeout = -1;

        cudaError_t error = cudaDeviceGetAttribute(&kernelExecTimeout, cudaDevAttrKernelExecTimeout, index);

        if (cudaSuccess != error)
            Utils::CudaHelper::PrintCudaError(error);

        return kernelExecTimeout;
    }

    bool Device::CanAccessPeer(int gpu1, int gpu2)
    {
        int canAccessPeer = -1;

        cudaError_t error = cudaDeviceCanAccessPeer(&canAccessPeer, gpu1, gpu2);

        if (cudaSuccess != error)
            Utils::CudaHelper::PrintCudaError(error);

        return canAccessPeer;
    }
}