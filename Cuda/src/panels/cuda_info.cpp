#include "cuda/panels/cuda_info.hpp"
#include "cuda/device.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "ui/widgets/texts/colored_text.hpp"
#include "ui/widgets/layouts/tree_node.hpp"
#include "utils/logs/logger.hpp"

namespace Cuda::Panels
{
    using UIColoredText    = UI::Widgets::Texts::ColoredText;
    using UIColoredTextPtr = std::shared_ptr<UIColoredText>;
    using UITreeNode       = UI::Widgets::Layouts::TreeNode;
    using UITreeNodePtr    = std::shared_ptr<UITreeNode>;

    CudaInfoPanel::CudaInfoPanel() : UI::Panels::WindowPanel("Cuda Info")
    {
        // Cuda::Utils::CudaHelper::PrintCudaInfo();

        int deviceCount = Cuda::Utils::CudaHelper::GetDeviceCount();

        // This function call returns 0 if there are no CUDA capable devices.
        if (!deviceCount)
        {
            ::Utils::Logs::Logger::Error("There are no available device(s) that support CUDA");
            return;
        }
        
        CreateWidget<UIColoredText>(UI::Types::Color::White, "There are %d CUDA Capable devices\n", deviceCount);
        
        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp properties = Cuda::Device::GetProperties(i);
            UITreeNodePtr gpu = CreateWidget<UITreeNode>(properties.name);
            
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Device %d: \"%s\"", i, properties.name);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "CUDA Driver Version / Runtime Version          %g / %g",
                Cuda::Utils::CudaHelper::GetDriverVersion(),
                Cuda::Utils::CudaHelper::GetRuntimeVersion());
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "CUDA Capability Major/Minor version number:    %d.%d", properties.major, properties.minor);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Total amount of global memory:                 %.0f MBytes (%llu bytes)",
                static_cast<float>(properties.totalGlobalMem / 1048576.0f),
                (unsigned long long)properties.totalGlobalMem);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "(%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores",
                properties.multiProcessorCount,
                Cuda::Utils::CudaHelper::ConvertSMVerToCores(properties.major, properties.minor),
                Cuda::Utils::CudaHelper::ConvertSMVerToCores(properties.major, properties.minor) * properties.multiProcessorCount);

            int clockRate = Cuda::Device::GetClockRate(i);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)",
                clockRate * 1e-3f, clockRate * 1e-6f);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Memory Clock rate:                             %.0f Mhz", Cuda::Device::GetMemoryClockRate(i));
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Memory Bus Width:                              %d-bit", properties.memoryBusWidth);
            if (properties.l2CacheSize)
                gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "L2 Cache Size:                                 %d bytes", properties.l2CacheSize);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)",
                properties.maxTexture1D,
                properties.maxTexture2D[0],
                properties.maxTexture2D[1],
                properties.maxTexture3D[0],
                properties.maxTexture3D[1],
                properties.maxTexture3D[2]);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers",
                properties.maxTexture1DLayered[0],
                properties.maxTexture1DLayered[1]);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers",
                properties.maxTexture2DLayered[0],
                properties.maxTexture2DLayered[1],
                properties.maxTexture2DLayered[2]);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Total amount of constant memory:               %zu bytes", properties.totalConstMem);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Total amount of shared memory per block:       %zu bytes", properties.sharedMemPerBlock);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Total shared memory per multiprocessor:        %zu bytes", properties.sharedMemPerMultiprocessor);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Total number of registers available per block: %d", properties.regsPerBlock);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Warp size:                                     %d", properties.warpSize);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Maximum number of threads per multiprocessor:  %d", properties.maxThreadsPerMultiProcessor);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Maximum number of threads per block:           %d", properties.maxThreadsPerBlock);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Max dimension size of a thread block (x,y,z): (%d, %d, %d)",
                properties.maxThreadsDim[0],
                properties.maxThreadsDim[1],
                properties.maxThreadsDim[2]);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Max dimension size of a grid size    (x,y,z): (%d, %d, %d)",
                properties.maxGridSize[0],
                properties.maxGridSize[1],
                properties.maxGridSize[2]);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Maximum memory pitch:                          %zu bytes", properties.memPitch);
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Texture alignment:                             %zu bytes", properties.textureAlignment);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Concurrent copy and kernel execution:          %s with %d copy engine(s)",
                (Cuda::Device::IsAttributeGpuOverlap(i) ? "Yes" : "No"),
                properties.asyncEngineCount);

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Run time limit on kernels:                     %s", Cuda::Device::IsKernelExecTimeout(i) ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Integrated GPU sharing Host Memory:            %s", properties.integrated ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Support host page-locked memory mapping:       %s", properties.canMapHostMemory ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Alignment requirement for Surfaces:            %s", properties.surfaceAlignment ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Device has ECC support:                        %s", properties.ECCEnabled ? "Enabled" : "Disabled");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Device supports Unified Addressing (UVA):      %s", properties.unifiedAddressing ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Device supports Managed Memory:                %s", properties.managedMemory ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Device supports Compute Preemption:            %s",
                properties.computePreemptionSupported ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Supports Cooperative Kernel Launch:            %s", properties.cooperativeLaunch ? "Yes" : "No");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d",
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

            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "Compute Mode:");
            gpu->CreateWidget<UIColoredText>(UI::Types::Color::White, "     < %s >", computeMode[Cuda::Device::GetComputeMode(i)].c_str());
        }
    } 
}
