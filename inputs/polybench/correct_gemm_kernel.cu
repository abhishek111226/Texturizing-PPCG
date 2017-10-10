#include <stdio.h> 
#define DEVICECODE true 
#include "correct_gemm_kernel.hu"
__global__ void kernel0(double *A, double *B, double *C, double alpha, double beta, int ni, int nj, int nk)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < ni; c0 += 8192)
      for (int c1 = 32 * b1; c1 < nj; c1 += 8192) {
        for (int c2 = 0; c2 < nk; c2 += 32) {
          if (ni >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, nj - c1 - 1); c4 += 16) {
              if (c2 == 0)
                C[(t0 + c0) * 4 + (c1 + c4)] *= beta;
              for (int c5 = 0; c5 <= ppcg_min(31, nk - c2 - 1); c5 += 1)
                C[(t0 + c0) * 4 + (c1 + c4)] += ((alpha * A[(t0 + c0) * 5 + (c2 + c5)]) * B[(c2 + c5) * 4 + (c1 + c4)]);
            }
          __syncthreads();
        }
        if (nk <= 0) {
          if (ni >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, nj - c1 - 1); c4 += 16)
              C[(t0 + c0) * 4 + (c1 + c4)] *= beta;
          __syncthreads();
        }
      }
}
