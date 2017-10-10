#include <stdio.h> 
#define DEVICECODE true 
#include "SeqAlign_kernel.hu"
__global__ void kernel0(float M[6000][4000])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    if (32 * b0 + t0 <= 5999) {
      if (b1 == 0) {
        if (32 * b0 + t0 >= 1 && t1 == 0) {
          M[32 * b0 + t0][0] = 0;
        } else if (b0 == 0 && t0 == 0)
          for (int c3 = t1; c3 <= 31; c3 += 16)
            M[0][c3] = 0;
      } else if (b0 == 0 && t0 == 0)
        for (int c3 = t1; c3 <= 31; c3 += 16)
          M[0][32 * b1 + c3] = 0;
    }
}

__device__ float max4(float a, float b, float c, float d)
{
    float max;
    if ( a > b ) {
        max = a;
    } else {
        max = b;
    }

    if ( c > max ) {
        max = c;
    }

    if ( d > max ) {
        max = d;
}
   return max;
}


__global__ void kernel1(float M[6000][4000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 >= 1 && c0 >= 32 * b0 + t0 + 1 && 32 * b0 + t0 + 5999 >= c0)
      M[-32 * b0 - t0 + c0][32 * b0 + t0] = max4(0, M[-32 * b0 - t0 + c0][32 * b0 + t0 - 1] - 8, M[-32 * b0 - t0 + c0 - 1][32 * b0 + t0] - 8, M[-32 * b0 - t0 + c0 - 1][32 * b0 + t0 - 1] + (tex2D(texRef_MatchQ, 32 * b0 + t0, -32 * b0 - t0 + c0)));
}
