#include <stdio.h> 
#define DEVICECODE true 
#include "3d_copy_kernel.hu"
__global__ void kernel0()
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 299)
      for (int c2 = 0; c2 <= 299; c2 += 32)
        for (int c4 = t1; c4 <= ppcg_min(31, -32 * b1 + 299); c4 += 4)
          for (int c5 = t2; c5 <= ppcg_min(31, -c2 + 299); c5 += 4)
            {
              int surf_write_temp_0 = ((tex3D(texRef_A, c2 + c5, 32 * b1 + c4, 32 * b0 + t0 - 1)));
              surf3Dwrite(surf_write_temp_0, surfRef_B,  (c2 + c5)  * sizeof(int),  (32 * b1 + c4) ,  (32 * b0 + t0) );
            };
}
