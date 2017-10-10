#include <stdio.h> 
#define DEVICECODE true 
#include "gesummv_kernel.hu"
__global__ void kernel0(float alpha, float beta, float tmp[1300], float x[1300], float y[1300])
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 0; c1 <= 1299; c1 += 32) {
      if (32 * b0 + t0 <= 1299 && c1 == 0)
        y[32 * b0 + t0] = 0.F;
      if (32 * b0 + t0 <= 1299) {
        for (int c3 = 0; c3 <= ppcg_min(31, -c1 + 1299); c3 += 1) {
          y[32 * b0 + t0] = (((tex2D(texRef_B, c1 + c3, 32 * b0 + t0)) * x[c1 + c3]) + y[32 * b0 + t0]);
          if (c1 == 0 && c3 == 0)
            tmp[32 * b0 + t0] = 0.F;
          tmp[32 * b0 + t0] = (((tex2D(texRef_A, c1 + c3, 32 * b0 + t0)) * x[c1 + c3]) + tmp[32 * b0 + t0]);
        }
        if (c1 == 1280)
          y[32 * b0 + t0] = ((alpha * tmp[32 * b0 + t0]) + (beta * y[32 * b0 + t0]));
      }
      __syncthreads();
    }
}
