#include <stdio.h> 
#define DEVICECODE true 
#include "chol_kernel.hu"
__global__ void kernel0(int A[3][3], int c0)
{

    {
      for (int c1 = 0; c1 < c0; c1 += 1)
        A[c0][c0] -= (A[c0][c1] * A[c0][c1]);
      if (c0 == 1)
        A[2][1] -= (A[2][0] * A[1][0]);
    }
}
__global__ void kernel1(int A[3][3], int c0)
{

    A[c0][c0] = A[c0][c0];
}
__global__ void kernel2(int A[3][3], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (t0 >= c0 + 1)
      A[t0][c0] /= A[c0][c0];
}
