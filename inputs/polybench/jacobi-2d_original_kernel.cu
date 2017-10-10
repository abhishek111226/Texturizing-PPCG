#include <stdio.h> 
#define DEVICECODE true 
#include "jacobi-2d_original_kernel.hu"
__global__ void kernel0(double *A, double *B, int tsteps, int n)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < n - 1; c0 += 8192)
      if (n >= t0 + c0 + 2 && t0 + c0 >= 1)
        for (int c3 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c3 <= ppcg_min(ppcg_min(31, n - 32 * b1 - 2), -32 * b1 + 1299); c3 += 16)
          B[(t0 + c0) * 1300 + (32 * b1 + c3)] = ((((((((((7 * (tex1Dfetch(texRef_A, (t0 + c0 - 1) * 1300 + (32 * b1 + c3 - 1)))) + (5 * (tex1Dfetch(texRef_A, (t0 + c0 - 1) * 1300 + (32 * b1 + c3))))) + (9 * (tex1Dfetch(texRef_A, (t0 + c0 - 1) * 1300 + (32 * b1 + c3 + 1))))) + (12 * (tex1Dfetch(texRef_A, (t0 + c0) * 1300 + (32 * b1 + c3 - 1))))) + (15 * (tex1Dfetch(texRef_A, (t0 + c0) * 1300 + (32 * b1 + c3))))) + (12 * (tex1Dfetch(texRef_A, (t0 + c0) * 1300 + (32 * b1 + c3 + 1))))) + (9 * (tex1Dfetch(texRef_A, (t0 + c0 + 1) * 1300 + (32 * b1 + c3 - 1))))) + (5 * (tex1Dfetch(texRef_A, (t0 + c0 + 1) * 1300 + (32 * b1 + c3))))) + (7 * (tex1Dfetch(texRef_A, (t0 + c0 + 1) * 1300 + (32 * b1 + c3 + 1))))) / 118);
}
