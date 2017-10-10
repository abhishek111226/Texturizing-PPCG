#include <stdio.h> 
#define DEVICECODE true 
#include "gramschmidt_kernel.hu"
__global__ void kernel0(float R[1200][1200])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c3 = ppcg_max(t1, 32 * b0 - 32 * b1 + t1 + 16 * ((t0 - t1 + 16) / 16)); c3 <= ppcg_min(31, -32 * b1 + 1199); c3 += 16)
      R[32 * b0 + t0][32 * b1 + c3] = 0.F;
}
__global__ void kernel1(float *nrm, int c0)
{

    nrm[0] = 0.F;
}
__global__ void kernel2(float A[1000][1200], float Q[1000][1200], float R[1200][1200], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_R[1][1];

    {
      if (t0 == 0)
        shared_R[0][0] = R[(c0 - 3597) / 5995][(c0 - 3597) / 5995];
      __syncthreads();
      if (32 * b0 + t0 <= 999)
        Q[32 * b0 + t0][(c0 - 3597) / 5995] = (A[32 * b0 + t0][(c0 - 3597) / 5995] / shared_R[0][0]);
    }
}
__global__ void kernel3(float A[1000][1200], float *nrm, int c0)
{
    float private_nrm;

    {
      private_nrm = *nrm;
      for (int c1 = 0; c1 <= 999; c1 += 1)
        private_nrm += (A[c1][(c0 - 1199) / 5995] * A[c1][(c0 - 1199) / 5995]);
      *nrm = private_nrm;
    }
}
__global__ void kernel4(float A[1000][1200], float Q[1000][1200], float R[1200][1200], int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (115104 * b0 + 3597 * t0 + 2398 >= c0 && 76736 * b0 + 2398 * t0 + 1439999 >= c0 && 2 * c0 >= 191840 * b0 + 5995 * t0 + 5995)
      for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 999); c4 += 16)
        A[32 * b1 + c4][(c0 / 1199) - 64 * b0 - 2 * t0 - 2] = (A[32 * b1 + c4][(c0 / 1199) - 64 * b0 - 2 * t0 - 2] - (Q[32 * b1 + c4][(-c0 / 1199) + 96 * b0 + 3 * t0 + 2] * R[(-c0 / 1199) + 96 * b0 + 3 * t0 + 2][(c0 / 1199) - 64 * b0 - 2 * t0 - 2]));
}
__global__ void kernel5(float A[1000][1200], float Q[1000][1200], float R[1200][1200], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (c0 >= 95920 * b0 + 2398 && 115104 * b0 + 112706 >= c0 && 76736 * b0 + 1513138 >= c0)
      for (int c2 = 0; c2 <= 992; c2 += 32) {
        if (115104 * b0 + 3597 * t0 + 1199 >= c0 && 76736 * b0 + 2398 * t0 + 1438800 >= c0 && 2 * c0 >= 191840 * b0 + 5995 * t0 + 3597)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 999); c4 += 1)
            R[(-c0 / 1199) + 96 * b0 + 3 * t0 + 1][(c0 / 1199) - 64 * b0 - 2 * t0 - 1] += (Q[c2 + c4][(-c0 / 1199) + 96 * b0 + 3 * t0 + 1] * A[c2 + c4][(c0 / 1199) - 64 * b0 - 2 * t0 - 1]);
        __syncthreads();
      }
}
__global__ void kernel6(float R[1200][1200], float *nrm, int c0)
{

    R[(c0 - 2398) / 5995][(c0 - 2398) / 5995] = sqrtf(nrm[0]);
}
