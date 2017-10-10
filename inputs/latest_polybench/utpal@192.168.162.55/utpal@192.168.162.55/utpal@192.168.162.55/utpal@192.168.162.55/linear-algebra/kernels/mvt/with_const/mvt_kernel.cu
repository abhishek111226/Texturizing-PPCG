#include <stdio.h> 
#define DEVICECODE true 
#include "mvt_kernel.hu"
__global__ void kernel0(float x1[2000], float y_1[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 1999; c1 += 32) {
      if (32 * b0 + t0 <= 1999)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1999); c3 += 1)
          x1[32 * b0 + t0] = (x1[32 * b0 + t0] + ((tex2D(texRef_A, c1 + c3, 32 * b0 + t0)) * y_1[c1 + c3]));
      __syncthreads();
    }
}
__global__ void kernel1(float x2[2000], float y_2[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 1999; c1 += 32) {
      if (32 * b0 + t0 <= 1999)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1999); c3 += 1)
          x2[32 * b0 + t0] = (x2[32 * b0 + t0] + ((tex2D(texRef_A, 32 * b0 + t0, c1 + c3)) * y_2[c1 + c3]));
      __syncthreads();
    }
}
