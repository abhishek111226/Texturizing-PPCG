#include <stdio.h> 
#define DEVICECODE true 
#include "trisolv_kernel.hu"
__global__ void kernel0(float b[2000], float x[2000])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 <= 1999)
      x[32 * b0 + t0] = b[32 * b0 + t0];
}
__global__ void kernel1(float L[2000][2000], float x[2000], int c0)
{

    x[c0 / 2] = (x[c0 / 2] / L[c0 / 2][c0 / 2]);
}
__global__ void kernel2(float L[2000][2000], float x[2000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_x_0[32];

    {
      if (t0 + c0 >= 32 * b0 + 31 && 32 * b0 + 2030 >= t0 + c0)
        shared_x_0[t0] = x[-32 * b0 + t0 + c0 - 31];
      __syncthreads();
      if (c0 >= 64 * b0 + 2 * t0 + 1 && 32 * b0 + t0 + 1999 >= c0)
        shared_x_0[-t0 + 31] -= (L[-32 * b0 - t0 + c0][32 * b0 + t0] * x[32 * b0 + t0]);
      __syncthreads();
      if (2 * t0 + c0 >= 64 * b0 + 63 && 32 * b0 + 2030 >= t0 + c0)
        x[-32 * b0 + t0 + c0 - 31] = shared_x_0[t0];
    }
}
