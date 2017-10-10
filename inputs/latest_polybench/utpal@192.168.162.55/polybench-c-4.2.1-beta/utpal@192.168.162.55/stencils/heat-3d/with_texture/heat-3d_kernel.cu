#include <stdio.h> 
#define DEVICECODE true 
#include "heat-3d_kernel.hu"
__global__ void kernel0(int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 118)
      for (int c3 = 0; c3 <= 118; c3 += 32)
        for (int c5 = ppcg_max(t1, ((t1 + 3) % 4) - 32 * b1 + 1); c5 <= ppcg_min(31, -32 * b1 + 118); c5 += 4)
          for (int c6 = ppcg_max(t2, ((t2 + c3 + 3) % 4) - c3 + 1); c6 <= ppcg_min(31, -c3 + 118); c6 += 4)
            {
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_0;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_0, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6wsw1B_1;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6wsw1B_1, surfRef_A, (c3 + c6 - 1) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_2;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_2, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6waw1B_3;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6waw1B_3, surfRef_A, (c3 + c6 + 1) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5wsw1BBc3wawc6B_4;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5wsw1BBc3wawc6B_4, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5 - 1, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_5;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_5, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5waw1BBc3wawc6B_6;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5waw1BBc3wawc6B_6, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5 + 1, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0wsw1BB32wmwb1wawc5BBc3wawc6B_7;
              surf3Dread(&surf_temp_AB32wmwb0wawt0wsw1BB32wmwb1wawc5BBc3wawc6B_7, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0 - 1);
              float surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_8;
              surf3Dread(&surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_8, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_AB32wmwb0wawt0waw1BB32wmwb1wawc5BBc3wawc6B_9;
              surf3Dread(&surf_temp_AB32wmwb0wawt0waw1BB32wmwb1wawc5BBc3wawc6B_9, surfRef_A, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0 + 1);
              
              {
                float surf_write_temp_0 = (((((0.125F * (((surf_temp_AB32wmwb0wawt0waw1BB32wmwb1wawc5BBc3wawc6B_9) - (2.F * (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_8))) + (surf_temp_AB32wmwb0wawt0wsw1BB32wmwb1wawc5BBc3wawc6B_7))) + (0.125F * (((surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5waw1BBc3wawc6B_6) - (2.F * (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_5))) + (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5wsw1BBc3wawc6B_4)))) + (0.125F * (((surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6waw1B_3) - (2.F * (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_2))) + (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6wsw1B_1)))) + (surf_temp_AB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_0)));
                surf3Dwrite(surf_write_temp_0, surfRef_B,  (c3 + c6)  * sizeof(float),  (32 * b1 + c5) ,  (32 * b0 + t0) );
              };
            }
}
__global__ void kernel1(int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.z, t1 = threadIdx.y, t2 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    if (32 * b0 + t0 >= 1 && 32 * b0 + t0 <= 118)
      for (int c3 = 0; c3 <= 118; c3 += 32)
        for (int c5 = ppcg_max(t1, ((t1 + 3) % 4) - 32 * b1 + 1); c5 <= ppcg_min(31, -32 * b1 + 118); c5 += 4)
          for (int c6 = ppcg_max(t2, ((t2 + c3 + 3) % 4) - c3 + 1); c6 <= ppcg_min(31, -c3 + 118); c6 += 4)
            {
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_0;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_0, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6wsw1B_1;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6wsw1B_1, surfRef_B, (c3 + c6 - 1) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_2;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_2, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6waw1B_3;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6waw1B_3, surfRef_B, (c3 + c6 + 1) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5wsw1BBc3wawc6B_4;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5wsw1BBc3wawc6B_4, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5 - 1, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_5;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_5, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5waw1BBc3wawc6B_6;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5waw1BBc3wawc6B_6, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5 + 1, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0wsw1BB32wmwb1wawc5BBc3wawc6B_7;
              surf3Dread(&surf_temp_BB32wmwb0wawt0wsw1BB32wmwb1wawc5BBc3wawc6B_7, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0 - 1);
              float surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_8;
              surf3Dread(&surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_8, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0);
              float surf_temp_BB32wmwb0wawt0waw1BB32wmwb1wawc5BBc3wawc6B_9;
              surf3Dread(&surf_temp_BB32wmwb0wawt0waw1BB32wmwb1wawc5BBc3wawc6B_9, surfRef_B, (c3 + c6) * sizeof(float), 32 * b1 + c5, 32 * b0 + t0 + 1);
              
              {
                float surf_write_temp_1 = (((((0.125F * (((surf_temp_BB32wmwb0wawt0waw1BB32wmwb1wawc5BBc3wawc6B_9) - (2.F * (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_8))) + (surf_temp_BB32wmwb0wawt0wsw1BB32wmwb1wawc5BBc3wawc6B_7))) + (0.125F * (((surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5waw1BBc3wawc6B_6) - (2.F * (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_5))) + (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5wsw1BBc3wawc6B_4)))) + (0.125F * (((surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6waw1B_3) - (2.F * (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_2))) + (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6wsw1B_1)))) + (surf_temp_BB32wmwb0wawt0BB32wmwb1wawc5BBc3wawc6B_0)));
                surf3Dwrite(surf_write_temp_1, surfRef_A,  (c3 + c6)  * sizeof(float),  (32 * b1 + c5) ,  (32 * b0 + t0) );
              };
            }
}
