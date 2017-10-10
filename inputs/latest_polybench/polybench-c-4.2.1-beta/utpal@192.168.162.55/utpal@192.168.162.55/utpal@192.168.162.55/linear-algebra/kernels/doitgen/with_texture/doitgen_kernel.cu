#include <stdio.h> 
#define DEVICECODE true 
#include "doitgen_kernel.hu"
__global__ void kernel0(float sum[160], int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    sum[32 * b0 + t0] = 0.F;
}
__global__ void kernel1(float sum[160], int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    for (int c3 = 0; c3 <= 159; c3 += 32) {
      for (int c5 = 0; c5 <= 31; c5 += 1)
        {
          float surf_temp_ABc0BBc1BBc3wawc5B_s1;
          surf3Dread(&surf_temp_ABc0BBc1BBc3wawc5B_s1, surfRef_A, (c3 + c5) * sizeof(float), c1, c0);
          
          sum[32 * b0 + t0] += ((surf_temp_ABc0BBc1BBc3wawc5B_s1) * (tex2D(texRef_C4, 32 * b0 + t0, c3 + c5)));
        }
      __syncthreads();
    }
}
__global__ void kernel2(float sum[160], int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    {
      float surf_write_temp_0 = (sum[32 * b0 + t0]);
      surf3Dwrite(surf_write_temp_0, surfRef_A,  (32 * b0 + t0)  * sizeof(float),  (c1) ,  (c0) );
    };
}
