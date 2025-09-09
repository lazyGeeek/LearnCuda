#pragma once
#ifndef CUDA_DEVICE_HPP_
#define CUDA_DEVICE_HPP_

#include <cuda_runtime.h>

namespace Cuda
{
    class Device
    {
    public:
        Device(const Device& other)             = delete;
        Device(Device&& other)                  = delete;
        Device& operator=(const Device& other)  = delete;
        Device& operator=(const Device&& other) = delete;

        static cudaDeviceProp GetProperties(int index);
        static int GetClockRate(int index);
        static int GetMemoryClockRate(int index);
        static int GetComputeMode(int index);

        static bool IsAttributeGpuOverlap(int index);
        static bool IsKernelExecTimeout(int index);
        static bool CanAccessPeer(int gpu1, int gpu2);
    };
}

#endif // CUDA_DEVICE_HPP_