#include "cuda/utils/cuda_helper.hpp"

namespace Cuda::Utils
{
    __global__ void FloatToColor(float* input, uint8_t* output)
    {
        // map from threadIdx/BlockIdx to pixel position
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = x + y * blockDim.x * gridDim.x;

        float l = input[offset];
        float s = 1.0f;
        int32_t h = (180 + static_cast<int32_t>(360.0f * input[offset])) % 360;
        float m1 = 0.0f;
        float m2 = 0.0f;

        if (l <= 0.5f)
            m2 = l * (1.0f + s);
        else
            m2 = l + s - l * s;
        m1 = 2.0f * l - m2;

        output[offset * 4 + 0] = Value(m1, m2, h + 120);
        output[offset * 4 + 1] = Value(m1, m2, h);
        output[offset * 4 + 2] = Value(m1, m2, h - 120);
        output[offset * 4 + 3] = 255;
    }

    __device__ uint8_t Value(float n1, float n2, int hue)
    {
        if (hue > 360.0f)    hue -= 360.0f;
        else if (hue < 0.0f) hue += 360.0f;

        if (hue < 60.0f)
            return static_cast<uint8_t>(255.0f * (n1 + (n2 - n1)* hue / 60.0f));
        if (hue < 180.0f)
            return static_cast<uint8_t>(255.0f * n2);
        if (hue < 240.0f)
            return static_cast<uint8_t>(255.0f * (n1 + (n2 - n1) * (240.0f - hue) / 60.0f));
        return static_cast<uint8_t>(255.0f * n1);
    }
}