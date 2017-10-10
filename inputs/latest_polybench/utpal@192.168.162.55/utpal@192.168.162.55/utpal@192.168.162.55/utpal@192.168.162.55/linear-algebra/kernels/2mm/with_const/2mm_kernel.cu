#include <stdio.h> 
#define DEVICECODE true 
#include "2mm_kernel.hu"
__global__ void kernel0(float A[800][1100], float alpha, float tmp[800][900])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 1099; c2 += 32) {
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 899); c4 += 16) {
        if (c2 == 0)
          tmp[32 * b0 + t0][32 * b1 + c4] = 0.F;
        for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 1099); c5 += 1)
          tmp[32 * b0 + t0][32 * b1 + c4] += ((alpha * A[32 * b0 + t0][c2 + c5]) * (tex2D(texRef_B, 32 * b1 + c4, c2 + c5)));
      }
      __syncthreads();
    }
}
__global__ void kernel1(float D[800][1200], float beta, float tmp[800][900])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 899; c2 += 32) {
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1199); c4 += 16) {
        if (c2 == 0)
          D[32 * b0 + t0][32 * b1 + c4] *= beta;
        for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 899); c5 += 1)
          D[32 * b0 + t0][32 * b1 + c4] += (tmp[32 * b0 + t0][c2 + c5] * (tex2D(texRef_C, 32 * b1 + c4, c2 + c5)));
      }
      __syncthreads();
    }
}
