#include <stdio.h> 
#define DEVICECODE true 
#include "SeqAlign_kernel.hu"
__global__ void kernel0(float M[3000][2000])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 <= 2999) {
      if (b1 == 0) {
        if (32 * b0 + t0 >= 1 && t1 == 0) {
          M[32 * b0 + t0][0] = 0;
        } else if (b0 == 0 && t0 == 0)
          for (int c3 = t1; c3 <= 31; c3 += 16)
            M[0][c3] = 0;
      } else if (b0 == 0 && t0 == 0)
        for (int c3 = t1; c3 <= ppcg_min(31, -32 * b1 + 1999); c3 += 16)
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


__global__ void kernel1(float M[3000][2000], float MatchQ[3000][2000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 1999 && c0 >= 32 * b0 + t0 + 1 && 32 * b0 + t0 + 2999 >= c0)
      M[-32 * b0 - t0 + c0][32 * b0 + t0] = max4(0, M[-32 * b0 - t0 + c0][32 * b0 + t0 - 1] - 8, M[-32 * b0 - t0 + c0 - 1][32 * b0 + t0] - 8, M[-32 * b0 - t0 + c0 - 1][32 * b0 + t0 - 1] + MatchQ[-32 * b0 - t0 + c0][32 * b0 + t0]);
}
