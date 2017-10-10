#include <stdio.h> 
#define DEVICECODE true 
#include "jacobi-1d_org_kernel.hu"
__global__ void kernel0(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0; c1 < n - 1; c1 += 1048576)
      if (n >= t0 + c1 + 2 && t0 + c1 >= 1)
        B[t0 + c1] = (0.33333 * (((tex1Dfetch(texRef_A, t0 + c1 - 1)) + (tex1Dfetch(texRef_A, t0 + c1))) + (tex1Dfetch(texRef_A, t0 + c1 + 1))));
}
__global__ void kernel1(double *A, double *B, int tsteps, int n, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0; c1 < n - 1; c1 += 1048576)
      if (n >= t0 + c1 + 2 && t0 + c1 >= 1)
        A[t0 + c1] = (0.33333 * (((tex1Dfetch(texRef_B, t0 + c1 - 1)) + (tex1Dfetch(texRef_B, t0 + c1))) + (tex1Dfetch(texRef_B, t0 + c1 + 1))));
}
