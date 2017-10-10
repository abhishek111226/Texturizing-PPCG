#include <stdio.h> 
#define DEVICECODE true 
#include "if_else_kernel.hu"
__global__ void kernel0(int *tex_array)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    if (t0 >= 1 && t1 >= 1)
      if ((tex1Dfetch(texRef_tex_array, t1 * 10 + t0)) == 0) {
        tex_array[t1 * 10 + t0] = (2 * t1);
      } else {
        if (((t1 + 1) % 3 >= 1 ? 0 : 1)) {
          tex_array[t1 * 10 + t0] = (2 * t1 + 1);
        } else {
          tex_array[t1 * 10 + t0] = (2 * t1 + 2);
        }
      }
}
