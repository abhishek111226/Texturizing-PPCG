#include <stdio.h> 
#define DEVICECODE true 
#include "jacobi-2d_texture_kernel.hu"
__global__ void kernel0(int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    for (int c1 = ((b0 + 255) % 256) + 1; c1 <= 1298; c1 += 256)
      for (int c2 = ((b1 + 255) % 256) + 1; c2 <= 1298; c2 += 256)
        {
          double surf_temp_ABc1wsw1BBc2B_0;
          surf2Dread(&surf_temp_ABc1wsw1BBc2B_0, surfRef_A, c2 * sizeof(double), c1 - 1);
          double surf_temp_ABc1waw1BBc2B_1;
          surf2Dread(&surf_temp_ABc1waw1BBc2B_1, surfRef_A, c2 * sizeof(double), c1 + 1);
          double surf_temp_ABc1BBc2waw1B_2;
          surf2Dread(&surf_temp_ABc1BBc2waw1B_2, surfRef_A, (c2 + 1) * sizeof(double), c1);
          double surf_temp_ABc1BBc2wsw1B_3;
          surf2Dread(&surf_temp_ABc1BBc2wsw1B_3, surfRef_A, (c2 - 1) * sizeof(double), c1);
          double surf_temp_ABc1BBc2B_4;
          surf2Dread(&surf_temp_ABc1BBc2B_4, surfRef_A, c2 * sizeof(double), c1);
          __threadfence_block();
          {
            double surf_write_temp_0 = ((((((surf_temp_ABc1BBc2B_4) + (surf_temp_ABc1BBc2wsw1B_3)) + (surf_temp_ABc1BBc2waw1B_2)) + (surf_temp_ABc1waw1BBc2B_1)) + (surf_temp_ABc1wsw1BBc2B_0)));
            surf2Dwrite(surf_write_temp_0, surfRef_B,  (c2)  * sizeof(double),  (c1) );
          };
        }
}
__global__ void kernel1(int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    for (int c1 = ((b0 + 255) % 256) + 1; c1 <= 1298; c1 += 256)
      for (int c2 = ((b1 + 255) % 256) + 1; c2 <= 1298; c2 += 256)
        {
          double surf_temp_BBc1wsw1BBc2B_0;
          surf2Dread(&surf_temp_BBc1wsw1BBc2B_0, surfRef_B, c2 * sizeof(double), c1 - 1);
          double surf_temp_BBc1waw1BBc2B_1;
          surf2Dread(&surf_temp_BBc1waw1BBc2B_1, surfRef_B, c2 * sizeof(double), c1 + 1);
          double surf_temp_BBc1BBc2waw1B_2;
          surf2Dread(&surf_temp_BBc1BBc2waw1B_2, surfRef_B, (c2 + 1) * sizeof(double), c1);
          double surf_temp_BBc1BBc2wsw1B_3;
          surf2Dread(&surf_temp_BBc1BBc2wsw1B_3, surfRef_B, (c2 - 1) * sizeof(double), c1);
          double surf_temp_BBc1BBc2B_4;
          surf2Dread(&surf_temp_BBc1BBc2B_4, surfRef_B, c2 * sizeof(double), c1);
          __threadfence_block();
          {
            double surf_write_temp_1 = ((((((surf_temp_BBc1BBc2B_4) + (surf_temp_BBc1BBc2wsw1B_3)) + (surf_temp_BBc1BBc2waw1B_2)) + (surf_temp_BBc1waw1BBc2B_1)) + (surf_temp_BBc1wsw1BBc2B_0)));
            surf2Dwrite(surf_write_temp_1, surfRef_A,  (c2)  * sizeof(double),  (c1) );
          };
        }
}
