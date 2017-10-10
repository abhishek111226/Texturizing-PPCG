#include <stdio.h> 
#define DEVICECODE true 
#include "poisson-3d_kernel.hu"
__global__ void kernel0(float B[119][120][120])
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 118)
      for (int c2 = 0; c2 <= 118; c2 += 32)
        for (int c4 = ppcg_max(t1, ((t1 + 3) % 4) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 118); c4 += 4)
          for (int c5 = ppcg_max(t2, ((t2 + c2 + 3) % 4) - c2 + 1); c5 <= ppcg_min(31, -c2 + 118); c5 += 4)
            B[32 * b0 + t0][32 * b1 + c4][c2 + c5] = (((2.666f * (tex3D(texRef_A, c2 + c5, 32 * b1 + c4, 32 * b0 + t0))) - (0.166f * ((((((tex3D(texRef_A, c2 + c5, 32 * b1 + c4, 32 * b0 + t0 - 1)) + (tex3D(texRef_A, c2 + c5, 32 * b1 + c4, 32 * b0 + t0 + 1))) + (tex3D(texRef_A, c2 + c5, 32 * b1 + c4 - 1, 32 * b0 + t0))) + (tex3D(texRef_A, c2 + c5, 32 * b1 + c4 + 1, 32 * b0 + t0))) + (tex3D(texRef_A, c2 + c5 + 1, 32 * b1 + c4, 32 * b0 + t0))) + (tex3D(texRef_A, c2 + c5 - 1, 32 * b1 + c4, 32 * b0 + t0))))) - (0.0833f * ((((((((((((tex3D(texRef_A, c2 + c5, 32 * b1 + c4 - 1, 32 * b0 + t0 - 1)) + (tex3D(texRef_A, c2 + c5, 32 * b1 + c4 - 1, 32 * b0 + t0 + 1))) + (tex3D(texRef_A, c2 + c5, 32 * b1 + c4 + 1, 32 * b0 + t0 - 1))) + (tex3D(texRef_A, c2 + c5, 32 * b1 + c4 + 1, 32 * b0 + t0 + 1))) + (tex3D(texRef_A, c2 + c5 - 1, 32 * b1 + c4, 32 * b0 + t0 - 1))) + (tex3D(texRef_A, c2 + c5 - 1, 32 * b1 + c4, 32 * b0 + t0 + 1))) + (tex3D(texRef_A, c2 + c5 - 1, 32 * b1 + c4 - 1, 32 * b0 + t0))) + (tex3D(texRef_A, c2 + c5 - 1, 32 * b1 + c4 + 1, 32 * b0 + t0))) + (tex3D(texRef_A, c2 + c5 + 1, 32 * b1 + c4, 32 * b0 + t0 - 1))) + (tex3D(texRef_A, c2 + c5 + 1, 32 * b1 + c4, 32 * b0 + t0 + 1))) + (tex3D(texRef_A, c2 + c5 + 1, 32 * b1 + c4 - 1, 32 * b0 + t0))) + (tex3D(texRef_A, c2 + c5 + 1, 32 * b1 + c4 + 1, 32 * b0 + t0)))));
}
