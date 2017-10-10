#include <stdio.h> 
#define DEVICECODE true 
#include "test_kernel.hu"
__global__ void kernel0(int *A, int *B)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;
    int private_A_0[1][1];

    if (t0 >= 1) {
      private_A_0[0][0] = (tex1Dfetch(texRef_A, (t0 - 1) * 4 + 0));
      B[t0 * 4 + 1] = private_A_0[0][0];
      A[t0 * 4 + 1] = private_A_0[0][0];
    }
}
