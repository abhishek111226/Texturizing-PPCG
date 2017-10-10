#include <stdio.h> 
#define DEVICECODE true 
#include "cholesky_correct_kernel.hu"
__global__ void kernel0(double *A, int n, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __assume(n<1000);
    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c1 = 32 * b0 + 8192 * ((-191872 * b0 + 3 * c0 + 48927356) / 49119232); c1 < ppcg_min((n + c0 - 3) / 1999, (2 * c0 - 3) / 3997); c1 += 8192)
      if (2 * c0 >= 3997 * t0 + 3997 * c1 + 4000 && 5996 * t0 + 5996 * c1 + 5999 >= 3 * c0)
        for (int c2 = ppcg_max(32 * b1 + 8192 * ((-191872 * b1 + c0 + 48939350) / 49119232), 32 * b1 + 8192 * ppcg_fdiv_q(-32 * b1 - c0 + 1999 * c1 + 1969, 8192) + 8192); c2 < n; c2 += 8192)
          for (int c4 = ppcg_max(t1, ((t0 + t1 + c0 + c1 + c2 + 15) % 16) + 1999 * t0 - c0 + 1999 * c1 - c2 + 2001); c4 <= ppcg_min(31, n - c2 - 1); c4 += 16)
            A[(c2 + c4) * 2000 + (1999 * t0 - c0 + 1999 * c1 + 2000)] = (A[(c2 + c4) * 2000 + (1999 * t0 - c0 + 1999 * c1 + 2000)] - (A[(c2 + c4) * 2000 + (-3997 * t0 + 2 * c0 - 3997 * c1 - 4000)] * A[(1999 * t0 - c0 + 1999 * c1 + 2000) * 2000 + (-3997 * t0 + 2 * c0 - 3997 * c1 - 4000)]));
}
/*__global__ void kernel1(double *A, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0 + 1048576 * ((-191904 * b0 + c0 + 6288128362) / 6288310272); c1 < n; c1 += 1048576)
      if (n >= t0 + c1 + 1 && 5997 * t0 + 5997 * c1 >= c0 + 3998)
        A[(t0 + c1) * 2000 + ((c0 - 1999) / 5997)] = (A[(t0 + c1) * 2000 + ((c0 - 1999) / 5997)] / A[((c0 - 1999) / 5997) * 2000 + ((c0 - 1999) / 5997)]);
}
__global__ void kernel2(double *A, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c1 = 32 * b0 + 1048576 * ((-191904 * b0 + c0 + 6288126363) / 6288310272); c1 <= ppcg_min(n - 1, c0 / 3998); c1 += 1048576)
      if (n >= t0 + c1 + 1 && c0 >= 3998 * t0 + 3998 * c1 && 5997 * t0 + 5997 * c1 >= c0 + 1999)
        A[(t0 + c1) * 2000 + (t0 + c1)] = (A[(t0 + c1) * 2000 + (t0 + c1)] - (A[(t0 + c1) * 2000 + ((c0 / 1999) - 2 * t0 - 2 * c1)] * A[(t0 + c1) * 2000 + ((c0 / 1999) - 2 * t0 - 2 * c1)]));
}
__global__ void kernel3(double *A, int n, int c0)
{

    A[(c0 / 5997) * 2000 + (c0 / 5997)] = sqrt(A[(c0 / 5997) * 2000 + (c0 / 5997)]);
}*/
