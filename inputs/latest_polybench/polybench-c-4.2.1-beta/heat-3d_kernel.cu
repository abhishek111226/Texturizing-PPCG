#include <stdio.h> 
#define DEVICECODE true 
#include "heat-3d_kernel.hu"
__global__ void kernel0(float A[200][200][200], float B[200][200][200], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 198)
      for (int c3 = 0; c3 <= 198; c3 += 32)
        for (int c5 = ppcg_max(t1, ((t1 + 3) % 4) - 32 * b1 + 1); c5 <= ppcg_min(31, -32 * b1 + 198); c5 += 4)
          for (int c6 = ppcg_max(t2, ((t2 + c3 + 3) % 4) - c3 + 1); c6 <= ppcg_min(31, -c3 + 198); c6 += 4)
            B[32 * b0 + t0][32 * b1 + c5][c3 + c6] = ((((0.125F * ((A[32 * b0 + t0 + 1][32 * b1 + c5][c3 + c6] - (2.F * A[32 * b0 + t0][32 * b1 + c5][c3 + c6])) + A[32 * b0 + t0 - 1][32 * b1 + c5][c3 + c6])) + (0.125F * ((A[32 * b0 + t0][32 * b1 + c5 + 1][c3 + c6] - (2.F * A[32 * b0 + t0][32 * b1 + c5][c3 + c6])) + A[32 * b0 + t0][32 * b1 + c5 - 1][c3 + c6]))) + (0.125F * ((A[32 * b0 + t0][32 * b1 + c5][c3 + c6 + 1] - (2.F * A[32 * b0 + t0][32 * b1 + c5][c3 + c6])) + A[32 * b0 + t0][32 * b1 + c5][c3 + c6 - 1]))) + A[32 * b0 + t0][32 * b1 + c5][c3 + c6]);
}
__global__ void kernel1(float A[200][200][200], float B[200][200][200], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 198)
      for (int c3 = 0; c3 <= 198; c3 += 32)
        for (int c5 = ppcg_max(t1, ((t1 + 3) % 4) - 32 * b1 + 1); c5 <= ppcg_min(31, -32 * b1 + 198); c5 += 4)
          for (int c6 = ppcg_max(t2, ((t2 + c3 + 3) % 4) - c3 + 1); c6 <= ppcg_min(31, -c3 + 198); c6 += 4)
            A[32 * b0 + t0][32 * b1 + c5][c3 + c6] = ((((0.125F * ((B[32 * b0 + t0 + 1][32 * b1 + c5][c3 + c6] - (2.F * B[32 * b0 + t0][32 * b1 + c5][c3 + c6])) + B[32 * b0 + t0 - 1][32 * b1 + c5][c3 + c6])) + (0.125F * ((B[32 * b0 + t0][32 * b1 + c5 + 1][c3 + c6] - (2.F * B[32 * b0 + t0][32 * b1 + c5][c3 + c6])) + B[32 * b0 + t0][32 * b1 + c5 - 1][c3 + c6]))) + (0.125F * ((B[32 * b0 + t0][32 * b1 + c5][c3 + c6 + 1] - (2.F * B[32 * b0 + t0][32 * b1 + c5][c3 + c6])) + B[32 * b0 + t0][32 * b1 + c5][c3 + c6 - 1]))) + B[32 * b0 + t0][32 * b1 + c5][c3 + c6]);
}
