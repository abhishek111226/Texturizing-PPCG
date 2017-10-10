#include <stdio.h> 
#define DEVICECODE true 
#include "gesummv_kernel.hu"
__global__ void kernel0(float A[1300][1300], float B[1300][1300], float alpha, float beta, float tmp[1300], float x[1300], float y[1300])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 1299; c1 += 32) {
      if (t0 + c1 <= 1299) {
        for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 1299); c2 += 1)
          shared_A[c2][t0] = A[32 * b0 + c2][t0 + c1];
        for (int c2 = 0; c2 <= ppcg_min(31, -32 * b0 + 1299); c2 += 1)
          shared_B[c2][t0] = B[32 * b0 + c2][t0 + c1];
      }
      __syncthreads();
      if (32 * b0 + t0 <= 1299 && c1 == 0)
        y[32 * b0 + t0] = 0.F;
      if (32 * b0 + t0 <= 1299) {
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1299); c3 += 1) {
          y[32 * b0 + t0] = ((shared_B[t0][c3] * x[c1 + c3]) + y[32 * b0 + t0]);
          if (c1 == 0 && c3 == 0)
            tmp[32 * b0 + t0] = 0.F;
          tmp[32 * b0 + t0] = ((shared_A[t0][c3] * x[c1 + c3]) + tmp[32 * b0 + t0]);
        }
        if (c1 == 1280)
          y[32 * b0 + t0] = ((alpha * tmp[32 * b0 + t0]) + (beta * y[32 * b0 + t0]));
      }
      __syncthreads();
    }
}
