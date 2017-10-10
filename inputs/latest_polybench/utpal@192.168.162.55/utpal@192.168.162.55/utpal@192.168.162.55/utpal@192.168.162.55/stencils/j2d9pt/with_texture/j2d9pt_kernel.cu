#include <stdio.h> 
#define DEVICECODE true 
#include "j2d9pt_kernel.hu"
__global__ void kernel0(float B[999][1000])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 998)
      for (int c3 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c3 <= ppcg_min(31, -32 * b1 + 998); c3 += 16)
        B[32 * b0 + t0][32 * b1 + c3] = ((((((((((7 * (tex2D(texRef_A, 32 * b1 + c3 - 1, 32 * b0 + t0 - 1))) + (5 * (tex2D(texRef_A, 32 * b1 + c3, 32 * b0 + t0 - 1)))) + (9 * (tex2D(texRef_A, 32 * b1 + c3 + 1, 32 * b0 + t0 - 1)))) + (12 * (tex2D(texRef_A, 32 * b1 + c3 - 1, 32 * b0 + t0)))) + (15 * (tex2D(texRef_A, 32 * b1 + c3, 32 * b0 + t0)))) + (12 * (tex2D(texRef_A, 32 * b1 + c3 + 1, 32 * b0 + t0)))) + (9 * (tex2D(texRef_A, 32 * b1 + c3 - 1, 32 * b0 + t0 + 1)))) + (5 * (tex2D(texRef_A, 32 * b1 + c3, 32 * b0 + t0 + 1)))) + (7 * (tex2D(texRef_A, 32 * b1 + c3 + 1, 32 * b0 + t0 + 1)))) / 118);
}
