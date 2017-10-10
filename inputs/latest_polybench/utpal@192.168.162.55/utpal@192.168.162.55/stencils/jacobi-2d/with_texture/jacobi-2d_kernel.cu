#include <stdio.h> 
#define DEVICECODE true 
#include "jacobi-2d_kernel.hu"
__global__ void kernel0(int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 1298)
      for (int c4 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 1298); c4 += 16)
        {
          float surf_temp_AB32wmwb0wawt0wsw1BB32wmwb1wawc4B_0;
          surf2Dread(&surf_temp_AB32wmwb0wawt0wsw1BB32wmwb1wawc4B_0, surfRef_A, (32 * b1 + c4) * sizeof(float), 32 * b0 + t0 - 1);
          float surf_temp_AB32wmwb0wawt0waw1BB32wmwb1wawc4B_1;
          surf2Dread(&surf_temp_AB32wmwb0wawt0waw1BB32wmwb1wawc4B_1, surfRef_A, (32 * b1 + c4) * sizeof(float), 32 * b0 + t0 + 1);
          float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4waw1B_2;
          surf2Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4waw1B_2, surfRef_A, (32 * b1 + c4 + 1) * sizeof(float), 32 * b0 + t0);
          float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4wsw1B_3;
          surf2Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4wsw1B_3, surfRef_A, (32 * b1 + c4 - 1) * sizeof(float), 32 * b0 + t0);
          float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4B_4;
          surf2Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4B_4, surfRef_A, (32 * b1 + c4) * sizeof(float), 32 * b0 + t0);
          
          {
            float surf_write_temp_0 = ((0.200000003F * (((((surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4B_4) + (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4wsw1B_3)) + (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc4waw1B_2)) + (surf_temp_AB32wmwb0wawt0waw1BB32wmwb1wawc4B_1)) + (surf_temp_AB32wmwb0wawt0wsw1BB32wmwb1wawc4B_0))));
            surf2Dwrite(surf_write_temp_0, surfRef_B,  (32 * b1 + c4)  * sizeof(float),  (32 * b0 + t0) );
          };
        }
}
__global__ void kernel1(int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 1298)
      for (int c4 = ppcg_max(t1, ((t1 + 15) % 16) - 32 * b1 + 1); c4 <= ppcg_min(31, -32 * b1 + 1298); c4 += 16)
        {
          float surf_temp_BB32wmwb0wawt0wsw1BB32wmwb1wawc4B_0;
          surf2Dread(&surf_temp_BB32wmwb0wawt0wsw1BB32wmwb1wawc4B_0, surfRef_B, (32 * b1 + c4) * sizeof(float), 32 * b0 + t0 - 1);
          float surf_temp_BB32wmwb0wawt0waw1BB32wmwb1wawc4B_1;
          surf2Dread(&surf_temp_BB32wmwb0wawt0waw1BB32wmwb1wawc4B_1, surfRef_B, (32 * b1 + c4) * sizeof(float), 32 * b0 + t0 + 1);
          float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4waw1B_2;
          surf2Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4waw1B_2, surfRef_B, (32 * b1 + c4 + 1) * sizeof(float), 32 * b0 + t0);
          float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4wsw1B_3;
          surf2Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4wsw1B_3, surfRef_B, (32 * b1 + c4 - 1) * sizeof(float), 32 * b0 + t0);
          float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4B_4;
          surf2Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4B_4, surfRef_B, (32 * b1 + c4) * sizeof(float), 32 * b0 + t0);
          
          {
            float surf_write_temp_1 = ((0.200000003F * (((((surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4B_4) + (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4wsw1B_3)) + (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc4waw1B_2)) + (surf_temp_BB32wmwb0wawt0waw1BB32wmwb1wawc4B_1)) + (surf_temp_BB32wmwb0wawt0wsw1BB32wmwb1wawc4B_0))));
            surf2Dwrite(surf_write_temp_1, surfRef_A,  (32 * b1 + c4)  * sizeof(float),  (32 * b0 + t0) );
          };
        }
}
