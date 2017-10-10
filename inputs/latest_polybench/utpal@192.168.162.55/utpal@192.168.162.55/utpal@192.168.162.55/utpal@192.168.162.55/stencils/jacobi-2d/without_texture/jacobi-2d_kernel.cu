#include <stdio.h> 
#define DEVICECODE true 
#include "jacobi-2d_kernel.hu"
__global__ void kernel0(float A[1300][1300], float B[1300][1300], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 1298)
      for (int c4 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 1298); c4 += 16)
        B[32 * b0 + t0][32 * b1 + c4] = (0.200000003F * ((((A[32 * b0 + t0][32 * b1 + c4] + A[32 * b0 + t0][32 * b1 + c4 - 1]) + A[32 * b0 + t0][32 * b1 + c4 + 1]) + A[32 * b0 + t0 + 1][32 * b1 + c4]) + A[32 * b0 + t0 - 1][32 * b1 + c4]));
}
__global__ void kernel1(float A[1300][1300], float B[1300][1300], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 1298)
      for (int c4 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 1298); c4 += 16)
        A[32 * b0 + t0][32 * b1 + c4] = (0.200000003F * ((((B[32 * b0 + t0][32 * b1 + c4] + B[32 * b0 + t0][32 * b1 + c4 - 1]) + B[32 * b0 + t0][32 * b1 + c4 + 1]) + B[32 * b0 + t0 + 1][32 * b1 + c4]) + B[32 * b0 + t0 - 1][32 * b1 + c4]));
}
