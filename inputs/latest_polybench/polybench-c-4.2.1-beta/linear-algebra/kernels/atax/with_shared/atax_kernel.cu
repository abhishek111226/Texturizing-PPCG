#include <stdio.h> 
#define DEVICECODE true 
#include "atax_kernel.hu"
__global__ void kernel0(float A[1900][2100], float tmp[1900], float x[2100])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_tmp[32];
    __shared__ float shared_x[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      for (int c1 = 0; c1 <= 2099; c1 += 32) {
        if (t0 + c1 <= 2099) {
          for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 1899); c2 += 1)
            shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
          shared_x[t0] = x[t0 + c1];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 1899 && c1 == 0)
          shared_tmp[t0] = 0.F;
        if (32 * b0 + t0 <= 1899)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 2099); c3 += 1)
            shared_tmp[t0] = (shared_tmp[t0] + (shared_A[t0][c3] * shared_x[c3]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 1899)
        tmp[32 * b0 + t0] = shared_tmp[t0];
    }
}
__global__ void kernel1(float A[1900][2100], float tmp[1900], float y[2100])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_tmp[32];
    __shared__ float shared_y[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      for (int c1 = 0; c1 <= 1899; c1 += 32) {
        if (t0 + c1 <= 1899)
          shared_tmp[t0] = tmp[t0 + c1];
        __syncthreads();
        if (32 * b0 + t0 <= 2099 && c1 == 0)
          shared_y[t0] = 0;
        if (32 * b0 + t0 <= 2099)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1899); c3 += 1)
            shared_y[t0] = (shared_y[t0] + (A[c1 + c3][32 * b0 + t0] * shared_tmp[c3]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 2099)
        y[32 * b0 + t0] = shared_y[t0];
    }
}
