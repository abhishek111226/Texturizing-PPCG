#include <stdio.h> 
#define DEVICECODE true 
#include "mvt_kernel.hu"
__global__ void kernel0(float x1[4000], float x2[4000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c2 = 0; c2 <= 3999; c2 += 32) {
      for (int c4 = 0; c4 <= 31; c4 += 1)
        x1[32 * b0 + t0] = (x1[32 * b0 + t0] + ((tex2D(texRef_A, c2 + c4, 32 * b0 + t0)) * x2[c2 + c4]));
      __syncthreads();
    }
}
__global__ void kernel1(float x1[4000], float x2[4000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c2 = 0; c2 <= 3999; c2 += 32) {
      for (int c4 = 0; c4 <= 31; c4 += 1)
        x2[32 * b0 + t0] = (x2[32 * b0 + t0] + ((tex2D(texRef_A, c2 + c4, 32 * b0 + t0)) * x1[c2 + c4]));
      __syncthreads();
    }
}
