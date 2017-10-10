#include <stdio.h> 
#define DEVICECODE true 
#include "rnn_kernel.hu"
__global__ void kernel0(float U[2000][1000], float inp_F[1000][1000], float s_F[1000][2000])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_U[32][32];
    __shared__ float shared_inp_F[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c2 = 0; c2 <= 999; c2 += 32) {
      if (32 * b1 + t0 <= 1999)
        for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 999); c4 += 16)
          shared_U[t0][c4] = U[32 * b1 + t0][c2 + c4];
      if (32 * b0 + t0 <= 999)
        for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 999); c4 += 16)
          shared_inp_F[t0][c4] = inp_F[32 * b0 + t0][c2 + c4];
      __syncthreads();
      if (32 * b0 + t0 <= 999)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1999); c4 += 16)
          for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 999); c5 += 1)
            s_F[32 * b0 + t0][32 * b1 + c4] += (shared_U[c4][c5] * shared_inp_F[t0][c5]);
      __syncthreads();
    }
}
__global__ void kernel1(float W[2000][2000], float s_F[1000][2000], int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_W[32][32];
    __shared__ float shared_s_F_0[1][32];
    __shared__ float shared_s_F_1[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 1999)
        shared_s_F_0[0][t0] = s_F[c0][32 * b0 + t0];
      for (int c2 = 0; c2 <= 1999; c2 += 32) {
        if (t0 + c2 <= 1999) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 1999); c3 += 1)
            shared_W[c3][t0] = W[32 * b0 + c3][t0 + c2];
          shared_s_F_1[0][t0] = s_F[c0 - 1][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 1999)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 1999); c4 += 1)
            shared_s_F_0[0][t0] += (shared_W[t0][c4] * shared_s_F_1[0][c4]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 1999)
        s_F[c0][32 * b0 + t0] = shared_s_F_0[0][t0];
    }
}
__global__ void kernel2(float V[1500][2000], float out_F[1000][1500], float s_F[1000][2000])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_V[32][32];
    __shared__ float shared_out_F[32][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (32 * b0 + t0 <= 999)
        for (int c1 = t1; c1 <= ppcg_min(31, -32 * b1 + 1499); c1 += 16)
          shared_out_F[t0][c1] = out_F[32 * b0 + t0][32 * b1 + c1];
      for (int c2 = 0; c2 <= 1999; c2 += 32) {
        if (32 * b1 + t0 <= 1499)
          for (int c4 = t1; c4 <= ppcg_min(31, -c2 + 1999); c4 += 16)
            shared_V[t0][c4] = V[32 * b1 + t0][c2 + c4];
        __syncthreads();
        if (32 * b0 + t0 <= 999)
          for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 1499); c4 += 16)
            for (int c5 = 0; c5 <= ppcg_min(31, -c2 + 1999); c5 += 1)
              shared_out_F[t0][c4] += (shared_V[c4][c5] * s_F[32 * b0 + t0][c2 + c5]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 999)
        for (int c1 = t1; c1 <= ppcg_min(31, -32 * b1 + 1499); c1 += 16)
          out_F[32 * b0 + t0][32 * b1 + c1] = shared_out_F[t0][c1];
    }
}
