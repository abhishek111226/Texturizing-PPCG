#include <stdio.h> 
#define DEVICECODE true 
#include "syr2k_org_kernel.hu"
__global__ void kernel0(double *A, double *B, double *C, double alpha, double beta, int n, int m)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * ((b0 - b1 + 256) % 256) + 32 * b1; c0 < n; c0 += 8192)
      for (int c1 = 32 * b1; c1 <= ppcg_min(n - 1, c0 + 31); c1 += 8192) {
        for (int c2 = 0; c2 < m; c2 += 32) {
          if (n >= t0 + c0 + 1) {
            if (c2 >= 32) {
              for (int c4 = t1; c4 <= ppcg_min(31, t0 + c0 - c1); c4 += 16)
                for (int c5 = 0; c5 <= ppcg_min(31, m - c2 - 1); c5 += 1)
                  C[(t0 + c0) * 1200 + (c1 + c4)] += ((((tex1Dfetch(texRef_A, (c1 + c4) * 1000 + (c2 + c5))) * alpha) * (tex1Dfetch(texRef_B, (t0 + c0) * 1000 + (c2 + c5)))) + (((tex1Dfetch(texRef_B, (c1 + c4) * 1000 + (c2 + c5))) * alpha) * (tex1Dfetch(texRef_A, (t0 + c0) * 1000 + (c2 + c5)))));
            } else
              for (int c4 = t1; c4 <= ppcg_min(31, t0 + c0 - c1); c4 += 16) {
                C[(t0 + c0) * 1200 + (c1 + c4)] *= beta;
                for (int c5 = 0; c5 <= ppcg_min(31, m - 1); c5 += 1)
                  C[(t0 + c0) * 1200 + (c1 + c4)] += ((((tex1Dfetch(texRef_A, (c1 + c4) * 1000 + c5)) * alpha) * (tex1Dfetch(texRef_B, (t0 + c0) * 1000 + c5))) + (((tex1Dfetch(texRef_B, (c1 + c4) * 1000 + c5)) * alpha) * (tex1Dfetch(texRef_A, (t0 + c0) * 1000 + c5))));
              }
          }
          __syncthreads();
        }
        if (m <= 0) {
          if (n >= t0 + c0 + 1)
            for (int c4 = t1; c4 <= ppcg_min(31, t0 + c0 - c1); c4 += 16)
              C[(t0 + c0) * 1200 + (c1 + c4)] *= beta;
          __syncthreads();
        }
      }
}
