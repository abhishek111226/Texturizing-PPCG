#include <stdio.h> 
#define DEVICECODE true 
#include "poisson-3d_kernel.hu"
__global__ void kernel0(float A[1000][1000][1000], float B[999][1000][1000])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 998)
      for (int c2 = 0; c2 <= 998; c2 += 32)
        for (int c4 = ppcg_max(t1, ((t1 + 3) % 4) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 998); c4 += 4)
          for (int c5 = ppcg_max(t2, ((t2 + c2 + 3) % 4) - c2 + 1); c5 <= ppcg_min(31, -c2 + 998); c5 += 4)
            B[32 * b0 + t0][32 * b1 + c4][c2 + c5] = (((2.666f * A[32 * b0 + t0][32 * b1 + c4][c2 + c5]) - (0.166f * (((((A[32 * b0 + t0 - 1][32 * b1 + c4][c2 + c5] + A[32 * b0 + t0 + 1][32 * b1 + c4][c2 + c5]) + A[32 * b0 + t0][32 * b1 + c4 - 1][c2 + c5]) + A[32 * b0 + t0][32 * b1 + c4 + 1][c2 + c5]) + A[32 * b0 + t0][32 * b1 + c4][c2 + c5 + 1]) + A[32 * b0 + t0][32 * b1 + c4][c2 + c5 - 1]))) - (0.0833f * (((((((((((A[32 * b0 + t0 - 1][32 * b1 + c4 - 1][c2 + c5] + A[32 * b0 + t0 + 1][32 * b1 + c4 - 1][c2 + c5]) + A[32 * b0 + t0 - 1][32 * b1 + c4 + 1][c2 + c5]) + A[32 * b0 + t0 + 1][32 * b1 + c4 + 1][c2 + c5]) + A[32 * b0 + t0 - 1][32 * b1 + c4][c2 + c5 - 1]) + A[32 * b0 + t0 + 1][32 * b1 + c4][c2 + c5 - 1]) + A[32 * b0 + t0][32 * b1 + c4 - 1][c2 + c5 - 1]) + A[32 * b0 + t0][32 * b1 + c4 + 1][c2 + c5 - 1]) + A[32 * b0 + t0 - 1][32 * b1 + c4][c2 + c5 + 1]) + A[32 * b0 + t0 + 1][32 * b1 + c4][c2 + c5 + 1]) + A[32 * b0 + t0][32 * b1 + c4 - 1][c2 + c5 + 1]) + A[32 * b0 + t0][32 * b1 + c4 + 1][c2 + c5 + 1])));
}
