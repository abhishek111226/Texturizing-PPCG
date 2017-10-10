#include <stdio.h> 
#define DEVICECODE true 
#include "mvt_kernel.hu"
__global__ void kernel0(float A[2000][2000], float x1[2000], float y_1[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_x1[32];
    __shared__ float shared_y_1[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 1999)
        shared_x1[t0] = x1[32 * b0 + t0];
      for (int c1 = 0; c1 <= 1999; c1 += 32) {
        if (t0 + c1 <= 1999) {
          for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 1999); c2 += 1)
            shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
          shared_y_1[t0] = y_1[t0 + c1];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 1999)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1999); c3 += 1)
            shared_x1[t0] = (shared_x1[t0] + (shared_A[t0][c3] * shared_y_1[c3]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 1999)
        x1[32 * b0 + t0] = shared_x1[t0];
    }
}
__global__ void kernel1(float A[2000][2000], float x2[2000], float y_2[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_x2[32];
    __shared__ float shared_y_2[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 1999)
        shared_x2[t0] = x2[32 * b0 + t0];
      __syncthreads();
      for (int c1 = 0; c1 <= 1999; c1 += 32) {
        if (t0 + c1 <= 1999)
          shared_y_2[t0] = y_2[t0 + c1];
        __syncthreads();
        if (32 * b0 + t0 <= 1999)
          for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1999); c3 += 1)
            shared_x2[t0] = (shared_x2[t0] + (A[c1 + c3][32 * b0 + t0] * shared_y_2[c3]));
        __syncthreads();
      }
      if (32 * b0 + t0 <= 1999)
        x2[32 * b0 + t0] = shared_x2[t0];
    }
}
