#include <stdio.h> 
#define DEVICECODE true 
#include "atax_kernel.hu"
__global__ void kernel0(float A[1900][2100], float tmp[1900], float x[2100])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 2099; c1 += 32) {
      if (32 * b0 + t0 <= 1899 && c1 == 0)
        tmp[32 * b0 + t0] = 0.F;
      if (32 * b0 + t0 <= 1899)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 2099); c3 += 1)
          tmp[32 * b0 + t0] = (tmp[32 * b0 + t0] + (A[32 * b0 + t0][c1 + c3] * x[c1 + c3]));
      __syncthreads();
    }
}
__global__ void kernel1(float A[1900][2100], float tmp[1900], float y[2100])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 1899; c1 += 32) {
      if (32 * b0 + t0 <= 2099 && c1 == 0)
        y[32 * b0 + t0] = 0;
      if (32 * b0 + t0 <= 2099)
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1899); c3 += 1)
          y[32 * b0 + t0] = (y[32 * b0 + t0] + (A[c1 + c3][32 * b0 + t0] * tmp[c1 + c3]));
      __syncthreads();
    }
}
