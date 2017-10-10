#include "stencil-2d_original_kernel.hu"
__global__ void kernel0(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < n - 1; c1 += 8192)
      if (n >= t0 + c1 + 2 && t0 + c1 >= 1)
        for (int c2 = 32 * b1; c2 < n - 1; c2 += 8192)
          for (int c4 = ppcg_max(t1, ((t1 + c2 + 15) % 16) - c2 + 1); c4 <= ppcg_min(31, n - c2 - 2); c4 += 16)
            B[(t0 + c1) * 1300 + (c2 + c4)] = ((((((((((7 * A[(t0 + c1 - 1) * 1300 + (c2 + c4 - 1)]) + (5 * A[(t0 + c1 - 1) * 1300 + (c2 + c4)])) + (9 * A[(t0 + c1 - 1) * 1300 + (c2 + c4 + 1)])) + (12 * A[(t0 + c1) * 1300 + (c2 + c4 - 1)])) + (15 * A[(t0 + c1) * 1300 + (c2 + c4)])) + (12 * A[(t0 + c1) * 1300 + (c2 + c4 + 1)])) + (9 * A[(t0 + c1 + 1) * 1300 + (c2 + c4 - 1)])) + (5 * A[(t0 + c1 + 1) * 1300 + (c2 + c4)])) + (7 * A[(t0 + c1 + 1) * 1300 + (c2 + c4 + 1)])) / 118);
}
__global__ void kernel1(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < n - 1; c1 += 8192)
      if (n >= t0 + c1 + 2 && t0 + c1 >= 1)
        for (int c2 = 32 * b1; c2 < n - 1; c2 += 8192)
          for (int c4 = ppcg_max(t1, ((t1 + c2 + 15) % 16) - c2 + 1); c4 <= ppcg_min(31, n - c2 - 2); c4 += 16)
            A[(t0 + c1) * 1300 + (c2 + c4)] = ((((((((((7 * B[(t0 + c1 - 1) * 1300 + (c2 + c4 - 1)]) + (5 * B[(t0 + c1 - 1) * 1300 + (c2 + c4)])) + (9 * B[(t0 + c1 - 1) * 1300 + (c2 + c4 + 1)])) + (12 * B[(t0 + c1) * 1300 + (c2 + c4 - 1)])) + (15 * B[(t0 + c1) * 1300 + (c2 + c4)])) + (12 * B[(t0 + c1) * 1300 + (c2 + c4 + 1)])) + (9 * B[(t0 + c1 + 1) * 1300 + (c2 + c4 - 1)])) + (5 * B[(t0 + c1 + 1) * 1300 + (c2 + c4)])) + (7 * B[(t0 + c1 + 1) * 1300 + (c2 + c4 + 1)])) / 118);
}
