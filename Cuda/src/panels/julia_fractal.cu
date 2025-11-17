#include "cuda/panels/julia_fractal.hpp"
#include "cuda/math/complex_number.hu"
#include "cuda/utils/cuda_helper.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/inputs/multiple_numbers_input.hpp"
#include "ui/widgets/inputs/single_number_input.hpp"
#include "ui/widgets/texts/text.hpp"
#include "utils/time/clock.hpp"

#include <cuda_runtime.h>

namespace Cuda::Panels
{
    __device__ int JuliaFractal::juliaGPU(int x, int y)
    {
        float jx = ((IMAGE_WIDTH / 2.0f - x) / (IMAGE_WIDTH / 2.0f)) / m_scale;
        float jy = ((IMAGE_HEIGHT / 2.0f - y) / (IMAGE_HEIGHT / 2.0f)) / m_scale;

        Cuda::Math::ComplexNum frac(m_constant.first, m_constant.second);
        Cuda::Math::ComplexNum curr(jx, jy);

        for (int i = 0; i < m_iterations; i++)
        {
            curr = curr * curr + frac;

            if (curr.GetMagnitude2() > ABSOLUTE_VALUE)
                return 0;
        }

        return 1;
    }

    __global__ void gpuStartCalculation(uint8_t* buffer, JuliaFractal& fractal)
    {
        int x = blockIdx.x;
        int y = blockIdx.y;
        int offset = x * 3 + y * 3 * gridDim.x;

        int color = 255 * fractal.juliaGPU(x, y);

        buffer[offset + 0] = color;
        buffer[offset + 1] = color;
        buffer[offset + 2] = color;
    }

    void JuliaFractal::calculateJuliaOnGPU()
    {
        if (!m_isGPUCalculationRunning)
            return;

        ::Utils::Time::Clock clock;
        clock.Start();

        uint8_t* buffer = m_gpuImageBuffer.data();
        cudaError_t error = cudaMalloc((void**)&buffer, m_gpuImageBuffer.size() * sizeof(uint8_t));

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        dim3 grid(IMAGE_WIDTH, IMAGE_HEIGHT);

        gpuStartCalculation<<<grid, 1>>>(buffer, *this);

        error = cudaMemcpy(m_gpuImageBuffer.data(), buffer, m_gpuImageBuffer.size(), cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        error = cudaFree(buffer);

        if (error != cudaSuccess)
            Cuda::Utils::CudaHelper::PrintCudaError(error);

        m_gpuCalculationTimeText->SetContent("Calculation time: %.3f milliseconds", clock.GetMicroseconds() / 1000.0f);
    }
}