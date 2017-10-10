#include <stdio.h> 
#define DEVICECODE true 
#include "bicg_kernel.hu"
__global__ void kernel0(float r[2100], float s[1900])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 2099; c1 += 32) {
      if (32 * b0 + t0 <= 1899 && c1 == 0)
        s[32 * b0 + t0] = 0;
      if (32 * b0 + t0 <= 1899)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 2099); c3 += 1)
          s[32 * b0 + t0] = (s[32 * b0 + t0] + (r[c1 + c3] * (tex2D(texRef_A, 32 * b0 + t0, c1 + c3))));
      __syncthreads();
    }
}
__global__ void kernel1(float p[1900], float q[2100])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 1899; c1 += 32) {
      if (32 * b0 + t0 <= 2099 && c1 == 0)
        q[32 * b0 + t0] = 0.F;
      if (32 * b0 + t0 <= 2099)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1899); c3 += 1)
          q[32 * b0 + t0] = (q[32 * b0 + t0] + ((tex2D(texRef_A, c1 + c3, 32 * b0 + t0)) * p[c1 + c3]));
      __syncthreads();
    }
}
