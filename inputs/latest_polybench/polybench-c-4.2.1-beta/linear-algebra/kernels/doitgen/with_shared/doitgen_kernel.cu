#include <stdio.h> 
#define DEVICECODE true 
#include "doitgen_kernel.hu"
__global__ void kernel0(float sum[160], int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    sum[32 * b0 + t0] = 0.F;
}
__global__ void kernel1(float A[150][140][160], float C4[160][160], float sum[160], int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[1][1][32];
    __shared__ float shared_sum[32];

    {
      shared_sum[t0] = sum[32 * b0 + t0];
      for (int c3 = 0; c3 <= 159; c3 += 32) {
        shared_A[0][0][t0] = A[c0][c1][t0 + c3];
        __syncthreads();
        for (int c5 = 0; c5 <= 31; c5 += 1)
          shared_sum[t0] += (shared_A[0][0][c5] * C4[c3 + c5][32 * b0 + t0]);
        __syncthreads();
      }
      sum[32 * b0 + t0] = shared_sum[t0];
    }
}
__global__ void kernel2(float A[150][140][160], float sum[160], int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    A[c0][c1][32 * b0 + t0] = sum[32 * b0 + t0];
}
