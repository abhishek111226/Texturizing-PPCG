#include <stdio.h> 
#define DEVICECODE true 
#include "lstm_kernel.hu"
__global__ void kernel0(float c_F[400][2850], float *o, float s_F[ns <= 0 ? 399 : 400][2850], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0; c1 < ns; c1 += 1048576)
      if (ns >= t0 + c1 + 1)
        s_F[c0 - 1][t0 + c1] = (c_F[c0 - 1][t0 + c1] * o[t0 + c1]);
}
__global__ void kernel1(float *i, int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 <= 2849)
      i[32 * b0 + t0] = 0.0;
}
__global__ void kernel2(float *f, int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 <= 2849)
      f[32 * b0 + t0] = 0.0;
}
__global__ void kernel3(float *g, int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 <= 2849)
      g[32 * b0 + t0] = 0.0;
}
__global__ void kernel4(float *o, int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    if (32 * b0 + t0 <= 2849)
      o[32 * b0 + t0] = 0.0;
}
__global__ void kernel5(float U_f[2850][3000], float *f, float inp_F[400][3000], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_U_f[32][32];
    __shared__ float shared_f[32];
    __shared__ float shared_inp_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_f[t0] = f[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_f[t0] = f[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2999; c2 += 32) {
        if (t0 + c2 <= 2999) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_U_f[c3][t0] = U_f[32 * b0 + c3][t0 + c2];
          shared_inp_F[0][t0] = inp_F[c0][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2999); c4 += 1)
            shared_f[t0] += (shared_U_f[t0][c4] * shared_inp_F[0][c4]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 2849 && c0 >= 1)
        f[32 * b0 + t0] = shared_f[t0];
    }
}
__global__ void kernel6(float U_g[2850][3000], float *g, float inp_F[400][3000], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_U_g[32][32];
    __shared__ float shared_g[32];
    __shared__ float shared_inp_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_g[t0] = g[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_g[t0] = g[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2999; c2 += 32) {
        if (t0 + c2 <= 2999) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_U_g[c3][t0] = U_g[32 * b0 + c3][t0 + c2];
          shared_inp_F[0][t0] = inp_F[c0][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2999); c4 += 1)
            shared_g[t0] += (shared_U_g[t0][c4] * shared_inp_F[0][c4]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 2849 && c0 >= 1)
        g[32 * b0 + t0] = shared_g[t0];
    }
}
__global__ void kernel7(float U_i[2850][3000], float *i, float inp_F[400][3000], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_U_i[32][32];
    __shared__ float shared_i[32];
    __shared__ float shared_inp_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_i[t0] = i[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_i[t0] = i[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2999; c2 += 32) {
        if (t0 + c2 <= 2999) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_U_i[c3][t0] = U_i[32 * b0 + c3][t0 + c2];
          shared_inp_F[0][t0] = inp_F[c0][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2999); c4 += 1)
            shared_i[t0] += (shared_U_i[t0][c4] * shared_inp_F[0][c4]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 2849 && c0 >= 1)
        i[32 * b0 + t0] = shared_i[t0];
    }
}
__global__ void kernel8(float W_f[2850][2850], float *f, float s_F[ns <= 0 ? 399 : 400][2850], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_W_f[32][32];
    __shared__ float shared_f[32];
    __shared__ float shared_s_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_f[t0] = f[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_f[t0] = f[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2849; c2 += 32) {
        if (t0 + c2 <= 2849) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_W_f[c3][t0] = W_f[32 * b0 + c3][t0 + c2];
          shared_s_F[0][t0] = s_F[c0 - 1][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2849); c4 += 1)
            shared_f[t0] += (shared_W_f[t0][c4] * shared_s_F[0][c4]);
        __syncthreads();
      }
      if (ns >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2849) {
        f[32 * b0 + t0] = shared_f[t0];
      } else if (32 * b0 + t0 >= ns && 32 * b0 + t0 <= 2849 && c0 == 399)
        f[32 * b0 + t0] = shared_f[t0];
    }
}
__global__ void kernel9(float U_o[2850][3000], float inp_F[400][3000], float *o, int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_U_o[32][32];
    __shared__ float shared_inp_F[1][32];
    __shared__ float shared_o[32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_o[t0] = o[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_o[t0] = o[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2999; c2 += 32) {
        if (t0 + c2 <= 2999) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_U_o[c3][t0] = U_o[32 * b0 + c3][t0 + c2];
          shared_inp_F[0][t0] = inp_F[c0][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2999); c4 += 1)
            shared_o[t0] += (shared_U_o[t0][c4] * shared_inp_F[0][c4]);
        __syncthreads();
      }
      if (32 * b0 + t0 <= 2849 && c0 >= 1) {
        o[32 * b0 + t0] = shared_o[t0];
      } else if (ns >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2849 && c0 == 0)
        o[32 * b0 + t0] = shared_o[t0];
    }
}
__global__ void kernel10(float W_i[2850][2850], float *i, float s_F[ns <= 0 ? 399 : 400][2850], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_W_i[32][32];
    __shared__ float shared_i[32];
    __shared__ float shared_s_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_i[t0] = i[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_i[t0] = i[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2849; c2 += 32) {
        if (t0 + c2 <= 2849) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_W_i[c3][t0] = W_i[32 * b0 + c3][t0 + c2];
          shared_s_F[0][t0] = s_F[c0 - 1][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2849); c4 += 1)
            shared_i[t0] += (shared_W_i[t0][c4] * shared_s_F[0][c4]);
        __syncthreads();
      }
      if (ns >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2849) {
        i[32 * b0 + t0] = shared_i[t0];
      } else if (32 * b0 + t0 >= ns && 32 * b0 + t0 <= 2849 && c0 == 399)
        i[32 * b0 + t0] = shared_i[t0];
    }
}
__global__ void kernel11(float W_g[2850][2850], float *g, float s_F[ns <= 0 ? 399 : 400][2850], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_W_g[32][32];
    __shared__ float shared_g[32];
    __shared__ float shared_s_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_g[t0] = g[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_g[t0] = g[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2849; c2 += 32) {
        if (t0 + c2 <= 2849) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_W_g[c3][t0] = W_g[32 * b0 + c3][t0 + c2];
          shared_s_F[0][t0] = s_F[c0 - 1][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2849); c4 += 1)
            shared_g[t0] += (shared_W_g[t0][c4] * shared_s_F[0][c4]);
        __syncthreads();
      }
      if (ns >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2849) {
        g[32 * b0 + t0] = shared_g[t0];
      } else if (32 * b0 + t0 >= ns && 32 * b0 + t0 <= 2849 && c0 == 399)
        g[32 * b0 + t0] = shared_g[t0];
    }
}
__global__ void kernel12(float W_o[2850][2850], float *o, float s_F[ns <= 0 ? 399 : 400][2850], int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    __shared__ float shared_W_o[32][32];
    __shared__ float shared_o[32];
    __shared__ float shared_s_F[1][32];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    {
      if (ns >= 2851 && ns >= 32 * b0 + t0 + 1) {
        shared_o[t0] = o[32 * b0 + t0];
      } else if (ns <= 2850 && 32 * b0 + t0 <= 2849)
        shared_o[t0] = o[32 * b0 + t0];
      for (int c2 = 0; c2 <= 2849; c2 += 32) {
        if (t0 + c2 <= 2849) {
          for (int c3 = 0; c3 <= ppcg_min(31, -32 * b0 + 2849); c3 += 1)
            shared_W_o[c3][t0] = W_o[32 * b0 + c3][t0 + c2];
          shared_s_F[0][t0] = s_F[c0 - 1][t0 + c2];
        }
        __syncthreads();
        if (32 * b0 + t0 <= 2849)
          for (int c4 = 0; c4 <= ppcg_min(31, -c2 + 2849); c4 += 1)
            shared_o[t0] += (shared_W_o[t0][c4] * shared_s_F[0][c4]);
        __syncthreads();
      }
      if (ns >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2849) {
        o[32 * b0 + t0] = shared_o[t0];
      } else if (32 * b0 + t0 >= ns && 32 * b0 + t0 <= 2849 && c0 == 399)
        o[32 * b0 + t0] = shared_o[t0];
    }
}
__global__ void kernel13(float c_F[400][2850], float *f, float *g, float *i, int ns, int c0)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c1 = 32 * b0; c1 < ns; c1 += 1048576)
      if (ns >= t0 + c1 + 1)
        c_F[c0][t0 + c1] = ((c_F[c0 - 1][t0 + c1] * f[t0 + c1]) + (g[t0 + c1] * i[t0 + c1]));
}
