#include "cuda/panels/heat.hpp"
#include "cuda/utils/cuda_helper.hpp"
#include "open_gl/resources/texture.hpp"
#include "ui/widgets/texts/text.hpp"

#include <cuda_runtime.h>

namespace Cuda::Panels
{
    __global__ void kernel(uchar4* output, cudaTextureObject_t texObj, int width, int height, Heat& heat)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        // Texture coordinates (normalized)
        float u = (x + 0.5f) / width;
        float v = (y + 0.5f) / height;
        
        float du = 1.0f / width;
        float dv = 1.0f / height;

        uchar4 t = tex2D<uchar4>(texObj, u, v + dv);
        uchar4 l = tex2D<uchar4>(texObj, u - du, v);
        uchar4 c = tex2D<uchar4>(texObj, u, v);
        uchar4 r = tex2D<uchar4>(texObj, u + du, v);
        uchar4 b = tex2D<uchar4>(texObj, u, v - dv);

        uchar4& res = output[y * width + x];
        res.x += c.x + heat.SPEED * ((t.x + b.x + r.x + l.x) - (4 * c.x));
        res.y += c.y + heat.SPEED * ((t.y + b.y + r.y + l.y) - (4 * c.y));
        res.z += c.z + heat.SPEED * ((t.z + b.z + r.z + l.z) - (4 * c.z));
        res.w += 255;

    }

    void Heat::calculate()
    {
        if (!m_isCalculationRunning.load(std::memory_order_acquire))
            return;

        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t cuArray;
        CUDA_HANDLE_ERROR(cudaMallocArray(&cuArray, &channelDesc, IMAGE_WIDTH, IMAGE_HEIGHT));
        CUDA_HANDLE_ERROR(cudaMemcpy2DToArray(cuArray, 0, 0, m_imageBuffer.data(), IMAGE_WIDTH * sizeof(uchar4), IMAGE_WIDTH * sizeof(uchar4), IMAGE_HEIGHT, cudaMemcpyHostToDevice));

        // Specify texture
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        // Create texture object
        cudaTextureObject_t texObj = 0;
        CUDA_HANDLE_ERROR(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        // Allocate result of transformation in device memory
        uchar4* output = nullptr;
        CUDA_HANDLE_ERROR(cudaMalloc(&output, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uchar4)));

        kernel<<<blocks, threads>>>(output, texObj, IMAGE_WIDTH, IMAGE_HEIGHT, *this);
        
        CUDA_HANDLE_ERROR(cudaMemcpy(m_imageBuffer.data(), output, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uchar4), cudaMemcpyDeviceToHost));
        CUDA_HANDLE_ERROR(cudaDestroyTextureObject(texObj));
        CUDA_HANDLE_ERROR(cudaFreeArray(cuArray));
        CUDA_HANDLE_ERROR(cudaFree(output));
    }
}