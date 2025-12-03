#pragma once
#ifndef CUDA_UTILS_CUDA_HELPER_HPP_
#define CUDA_UTILS_CUDA_HELPER_HPP_

#include <cuda_runtime.h>
#include <cuda.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace Cuda::Utils
{
    #define CUDA_HANDLE_ERROR(error) {\
            if (error != cudaSuccess)\
            {\
                Cuda::Utils::CudaHelper::PrintCudaError(error);\
                throw std::runtime_error("ERROR: Cuda error");\
            }\
        };

    struct MtoCores
    {
        // 0xMm (hexidecimal notation),
        // M = SM Major version, and m = SM minor version
        int SM;
        int Cores;
    };

    __global__ void FloatToColor(float* input, uint8_t* output);
    __device__ uint8_t Value(float n1, float n2, int hue);

    class CudaHelper
    {
    public:
        static int ConvertSMVerToCores(int major, int minor);
        
        static int GetDeviceCount();
        static double GetDriverVersion();
        static double GetRuntimeVersion();
        
        static void PrintCudaInfo();
        static void PrintCudaError(cudaError_t error);
        
    private:
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