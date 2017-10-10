#include <stdio.h> 
#define DEVICECODE true 
#include "mvt_kernel.hu"
__global__ void kernel0(float A[4000][4000], float x1[4000], float x2[4000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_x1[32];
    __shared__ float shared_x2[32];

    {
      shared_x1[t0] = x1[32 * b0 + t0];
      for (int c2 = 0; c2 <= 3999; c2 += 32) {
        for (int c3 = 0; c3 <= 31; c3 += 1)
          shared_A[c3][t0] = A[32 * b0 + c3][t0 + c2];
        shared_x2[t0] = x2[t0 + c2];
        __syncthreads();
        for (int c4 = 0; c4 <= 31; c4 += 1)
          shared_x1[t0] = (shared_x1[t0] + (shared_A[t0][c4] * shared_x2[c4]));
        __syncthreads();
      }
      x1[32 * b0 + t0] = shared_x1[t0];
    }
}
__global__ void kernel1(float A[4000][4000], float x1[4000], float x2[4000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_x1[32];
    __shared__ float shared_x2[32];

    {
      shared_x2[t0] = x2[32 * b0 + t0];
      for (int c2 = 0; c2 <= 3999; c2 += 32) {
        for (int c3 = 0; c3 <= 31; c3 += 1)
          shared_A[c3][t0] = A[32 * b0 + c3][t0 + c2];
        shared_x1[t0] = x1[t0 + c2];
        __syncthreads();
        for (int c4 = 0; c4 <= 31; c4 += 1)
          shared_x2[t0] = (shared_x2[t0] + (shared_A[t0][c4] * shared_x1[c4]));
        __syncthreads();
      }
      x2[32 * b0 + t0] = shared_x2[t0];
    }
}
