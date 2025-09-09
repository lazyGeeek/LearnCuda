#include "learn_cuda/application.hpp"
#include "cuda/device.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "utils/logs/logger.hpp"

#include <string>
#include <vector>

namespace LearnCuda
{
    Application::Application(const std::filesystem::path& projectPath) :
        ProjectPath(projectPath)
    {
    }

    Application::~Application()
    {
    }

    void Application::Run()
    {
        int deviceCount = Cuda::Utils::CudaHelper::GetDeviceCount();

        // This function call returns 0 if there are no CUDA capable devices.
        if (!deviceCount)
        {
            ::Utils::Logs::Logger::Error("There are no available device(s) that support CUDA");
            return;
        }

        ::Utils::Logs::Logger::Info("There are %d CUDA Capable devices\n", deviceCount);

        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp properties = Cuda::Device::GetProperties(i);

            ::Utils::Logs::Logger::Info("Device %d: \"%s\"", i, properties.name);

            // Console log
            ::Utils::Logs::Logger::Info("CUDA Driver Version / Runtime Version          %g / %g",
                Cuda::Utils::CudaHelper::GetDriverVersion(),
                Cuda::Utils::CudaHelper::GetRuntimeVersion());
            ::Utils::Logs::Logger::Info("CUDA Capability Major/Minor version number:    %d.%d", properties.major, properties.minor);

            ::Utils::Logs::Logger::Info("Total amount of global memory:                 %.0f MBytes (%llu bytes)",
                static_cast<float>(properties.totalGlobalMem / 1048576.0f),
                (unsigned long long)properties.totalGlobalMem);

            ::Utils::Logs::Logger::Info("(%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores",
                properties.multiProcessorCount,
                Cuda::Utils::CudaHelper::ConvertSMVerToCores(properties.major, properties.minor),
                Cuda::Utils::CudaHelper::ConvertSMVerToCores(properties.major, properties.minor) * properties.multiProcessorCount);

            int clockRate = Cuda::Device::GetClockRate(i);
            ::Utils::Logs::Logger::Info("GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)",
                clockRate * 1e-3f, clockRate * 1e-6f);

            ::Utils::Logs::Logger::Info("Memory Clock rate:                             %.0f Mhz", Cuda::Device::GetMemoryClockRate(i));
            ::Utils::Logs::Logger::Info("Memory Bus Width:                              %d-bit", properties.memoryBusWidth);
            if (properties.l2CacheSize)
                ::Utils::Logs::Logger::Info("L2 Cache Size:                                 %d bytes", properties.l2CacheSize);

            ::Utils::Logs::Logger::Info("Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)",
                properties.maxTexture1D,
                properties.maxTexture2D[0],
                properties.maxTexture2D[1],
                properties.maxTexture3D[0],
                properties.maxTexture3D[1],
                properties.maxTexture3D[2]);
            ::Utils::Logs::Logger::Info("Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers",
                properties.maxTexture1DLayered[0],
                properties.maxTexture1DLayered[1]);
            ::Utils::Logs::Logger::Info("Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers",
                properties.maxTexture2DLayered[0],
                properties.maxTexture2DLayered[1],
                properties.maxTexture2DLayered[2]);

            ::Utils::Logs::Logger::Info("Total amount of constant memory:               %zu bytes", properties.totalConstMem);
            ::Utils::Logs::Logger::Info("Total amount of shared memory per block:       %zu bytes", properties.sharedMemPerBlock);
            ::Utils::Logs::Logger::Info("Total shared memory per multiprocessor:        %zu bytes", properties.sharedMemPerMultiprocessor);
            ::Utils::Logs::Logger::Info("Total number of registers available per block: %d", properties.regsPerBlock);
            ::Utils::Logs::Logger::Info("Warp size:                                     %d", properties.warpSize);
            ::Utils::Logs::Logger::Info("Maximum number of threads per multiprocessor:  %d", properties.maxThreadsPerMultiProcessor);
            ::Utils::Logs::Logger::Info("Maximum number of threads per block:           %d", properties.maxThreadsPerBlock);
            ::Utils::Logs::Logger::Info("Max dimension size of a thread block (x,y,z): (%d, %d, %d)",
                properties.maxThreadsDim[0],
                properties.maxThreadsDim[1],
                properties.maxThreadsDim[2]);
            ::Utils::Logs::Logger::Info("Max dimension size of a grid size    (x,y,z): (%d, %d, %d)",
                properties.maxGridSize[0],
                properties.maxGridSize[1],
                properties.maxGridSize[2]);
            ::Utils::Logs::Logger::Info("Maximum memory pitch:                          %zu bytes", properties.memPitch);
            ::Utils::Logs::Logger::Info("Texture alignment:                             %zu bytes", properties.textureAlignment);

            ::Utils::Logs::Logger::Info("Concurrent copy and kernel execution:          %s with %d copy engine(s)",
                (Cuda::Device::IsAttributeGpuOverlap(i) ? "Yes" : "No"),
                properties.asyncEngineCount);

            ::Utils::Logs::Logger::Info("Run time limit on kernels:                     %s", Cuda::Device::IsKernelExecTimeout(i) ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Integrated GPU sharing Host Memory:            %s", properties.integrated ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Support host page-locked memory mapping:       %s", properties.canMapHostMemory ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Alignment requirement for Surfaces:            %s", properties.surfaceAlignment ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Device has ECC support:                        %s", properties.ECCEnabled ? "Enabled" : "Disabled");
            ::Utils::Logs::Logger::Info("Device supports Unified Addressing (UVA):      %s", properties.unifiedAddressing ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Device supports Managed Memory:                %s", properties.managedMemory ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Device supports Compute Preemption:            %s",
                properties.computePreemptionSupported ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Supports Cooperative Kernel Launch:            %s", properties.cooperativeLaunch ? "Yes" : "No");
            ::Utils::Logs::Logger::Info("Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d",
                properties.pciDomainID, properties.pciBusID, properties.pciDeviceID);

            std::vector<std::string> computeMode =
            {
                "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
                "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
                "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
                "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
                "Unknown",
                ""
            };

            ::Utils::Logs::Logger::Info("Compute Mode:");
            ::Utils::Logs::Logger::Info("     < %s >", computeMode[Cuda::Device::GetComputeMode(i)].c_str());

            Cuda::Utils::CudaHelper::PrintCudaInfo();
        }
    }
}
