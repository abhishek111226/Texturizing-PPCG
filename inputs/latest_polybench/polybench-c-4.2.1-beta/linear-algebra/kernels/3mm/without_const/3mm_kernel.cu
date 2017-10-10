#include <stdio.h> 
#define DEVICECODE true 
#include "3mm_kernel.hu"
__global__ void kernel0(float A[800][1000], float B[1000][900], float E[800][900])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 999; c2 += 32) {
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 899); c4 += 16) {
        if (c2 == 0)
          E[32 * b0 + t0][32 * b1 + c4] = 0.F;
        for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 999); c5 += 1)
          E[32 * b0 + t0][32 * b1 + c4] += (A[32 * b0 + t0][c2 + c5] * B[c2 + c5][32 * b1 + c4]);
      }
      __syncthreads();
    }
}
__global__ void kernel1(float C[900][1200], float D[1200][1100], float F[900][1100])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 1199; c2 += 32) {
      if (32 * b0 + t0 <= 899)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1099); c4 += 16) {
          if (c2 == 0)
            F[32 * b0 + t0][32 * b1 + c4] = 0.F;
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 1199); c5 += 1)
            F[32 * b0 + t0][32 * b1 + c4] += (C[32 * b0 + t0][c2 + c5] * D[c2 + c5][32 * b1 + c4]);
        }
      __syncthreads();
    }
}
__global__ void kernel2(float E[800][900], float F[900][1100], float G[800][1100])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 899; c2 += 32) {
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1099); c4 += 16) {
        if (c2 == 0)
          G[32 * b0 + t0][32 * b1 + c4] = 0.F;
        for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 899); c5 += 1)
          G[32 * b0 + t0][32 * b1 + c4] += (E[32 * b0 + t0][c2 + c5] * F[c2 + c5][32 * b1 + c4]);
      }
      __syncthreads();
    }
}
