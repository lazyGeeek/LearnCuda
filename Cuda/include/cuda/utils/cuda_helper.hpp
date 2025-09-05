#pragma once
#ifndef CUDA_UTILS_CUDA_HELPER_HPP_
#define CUDA_UTILS_CUDA_HELPER_HPP_

#include <cuda_runtime.h>
#include <cuda.h>

#include <string>
#include <vector>

namespace Cuda::Utils
{
    struct MtoCores
    {
        // 0xMm (hexidecimal notation),
        // M = SM Major version, and m = SM minor version
        int SM;
        int Cores;
    };

    class CudaHelper
    {
    public:
        static int ConvertSMVerToCores(int major, int minor);
        
        static int GetDeviceCount();
        static int GetDriverVersion();
        static int GetRuntimeVersion();
        
        static void PrintCudaInfo();
        
    private:
        static void printCudaError(cudaError_t error);
        static std::string cudaGetErrorEnum(cudaError_t error);
        static std::string cudaGetErrorEnum(CUresult error)
        {
            const char *ret = NULL;
            cuGetErrorName(error, &ret);
            return ret ? ret : "<unknown>";
        }
            
        static std::vector<MtoCores> m_gpuArchCoresPerSM;
    };
}

#endif // CUDA_UTILS_CUDA_HELPER_HPP_