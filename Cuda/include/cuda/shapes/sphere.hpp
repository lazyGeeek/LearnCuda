#pragma once
#ifndef CUDA_UTILS_SPHERE_HPP_
#define CUDA_UTILS_SPHERE_HPP_

#include <cuda_runtime.h>

#include <cmath>
#include <limits.h>

namespace Cuda::Shapes
{
    struct Sphere
    {
        float R, G, B;
        float Radius;
        float X, Y, Z;

        __device__ float Hit(float ox, float oy, float& n)
        {
            float dx = ox - X;
            float dy = oy - Y;

            if (dx * dx + dy * dy < Radius * Radius)
            {
                float dz = sqrt(Radius * Radius - dx * dx - dy * dy);
                n = dz / sqrt(Radius * Radius);
                return dz + Z;
            }
            
            return FLT_MIN;
        }
    };
}

#endif // CUDA_UTILS_SPHERE_HPP_