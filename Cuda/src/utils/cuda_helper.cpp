#include "cuda/utils/cuda_helper.hpp"
#include "utils/logs/logger.hpp"

#include <sstream>
// #include <stdexcept>

namespace Cuda::Utils
{
    std::vector<MtoCores> CudaHelper::m_gpuArchCoresPerSM
    {
        { 0x30, 192 },
        { 0x32, 192 },
        { 0x35, 192 },
        { 0x37, 192 },
        { 0x50, 128 },
        { 0x52, 128 },
        { 0x53, 128 },
        { 0x60,  64 },
        { 0x61, 128 },
        { 0x62, 128 },
        { 0x70,  64 },
        { 0x72,  64 },
        { 0x75,  64 },
        { 0x80,  64 },
        { 0x86, 128 },
        { 0x87, 128 },
        { 0x89, 128 },
        { 0x90, 128 },
        { 0xa0, 128 },
        { 0xa1, 128 },
        { 0xa3, 128 },
        { 0xb0, 128 },
        { 0xc0, 128 },
        { 0xc1, 128 },
        { -1,   -1  }
    };

    int CudaHelper::ConvertSMVerToCores(int major, int minor)
    {
        int index = 0;

        while (m_gpuArchCoresPerSM[index].SM != -1)
        {
            if (m_gpuArchCoresPerSM[index].SM == ((major << 4) + minor))
                return m_gpuArchCoresPerSM[index].Cores;

            index++;
        }

        ::Utils::Logs::Logger::Error("MapSMtoCores for SM %d.%d is undefined. "
                                     "Default to use %d Cores/SM\n",
                                     major, minor, m_gpuArchCoresPerSM[index - 1].Cores);

        return m_gpuArchCoresPerSM[index - 1].Cores;
    }

    std::string CudaHelper::cudaGetErrorEnum(cudaError_t error)
    {
        return cudaGetErrorName(error);
    }

    void CudaHelper::printCudaError(cudaError_t error)
    {
        if (cudaSuccess != error)
        {
            ::Utils::Logs::Logger::Error("CUDA Error: %s: %i\nCode: %d; Description: %s\n",
                __FILE__, __LINE__, static_cast<int>(error), cudaGetErrorString(error));
        }
    }

    int CudaHelper::GetDeviceCount()
    {
        int deviceCount = 0;
        cudaError_t errorId = cudaGetDeviceCount(&deviceCount);

        if (errorId != cudaError::cudaSuccess)
            printCudaError(errorId);

        return deviceCount;
    }

    int CudaHelper::GetDriverVersion()
    {
        int vesrion = 0;

        cudaError_t errorId = cudaDriverGetVersion(&vesrion);

        if (errorId != cudaError::cudaSuccess)
            printCudaError(errorId);

        return vesrion;
    }
    
    int CudaHelper::GetRuntimeVersion()
    {
        int vesrion = 0;

        cudaError_t errorId = cudaRuntimeGetVersion(&vesrion);

        if (errorId != cudaError::cudaSuccess)
            printCudaError(errorId);

        return vesrion;
    }

    void CudaHelper::PrintCudaInfo()
    {
        ::Utils::Logs::Logger::Info("Driver = CUDART, Driver Version = %d, Runtime Version = %d, NumDevs = %d;",
            GetDriverVersion(), GetRuntimeVersion(), GetDeviceCount());
    }
}