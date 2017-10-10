#include <stdio.h> 
#define DEVICECODE true 
#include "gesummv_org_kernel.hu"
__global__ void kernel0(double *A, double *B, double alpha, double beta, double *tmp, double *x, double *y, int n)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n; c0 += 1048576)
      for (int c1 = 0; c1 <= n; c1 += 32) {
        if (n >= t0 + c0 + 1 && c1 == 0)
          y[t0 + c0] = 0.;
        if (n >= t0 + c0 + 1) {
          for (int c3 = 0; c3 <= ppcg_min(31, n - c1 - 1); c3 += 1) {
            y[t0 + c0] = (((tex1Dfetch(texRef_B, (t0 + c0) * 1300 + (c1 + c3))) * (tex1Dfetch(texRef_x, c1 + c3))) + y[t0 + c0]);
            if (c1 == 0 && c3 == 0)
              tmp[t0 + c0] = 0.;
            tmp[t0 + c0] = (((tex1Dfetch(texRef_A, (t0 + c0) * 1300 + (c1 + c3))) * (tex1Dfetch(texRef_x, c1 + c3))) + tmp[t0 + c0]);
          }
          if (c1 + 31 >= n)
            y[t0 + c0] = ((alpha * tmp[t0 + c0]) + (beta * y[t0 + c0]));
        }
        __syncthreads();
      }
}
