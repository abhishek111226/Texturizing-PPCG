#include <stdio.h> 
#define DEVICECODE true 
#include "floyd-warshall_org_kernel.hu"
__global__ void kernel0(int *path, int n, int c0, int c1)
{
    int b0 = blockIdx.x;
    int t0 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    #define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
    for (int c2 = ppcg_max(32 * b0, 32 * b0 + 1048576 * ppcg_fdiv_q(-n - 32 * b0 + c1 - 31, 1048576) + 1048576); c2 <= ppcg_min(n - 1, c1); c2 += 1048576)
      if (n >= t0 + c2 + 1 && n + t0 + c2 >= c1 + 1 && c1 >= t0 + c2)
        path[(-t0 + c1 - c2) * 2800 + (t0 + c2)] = ((path[(-t0 + c1 - c2) * 2800 + (t0 + c2)] < (path[(-t0 + c1 - c2) * 2800 + c0] + path[c0 * 2800 + (t0 + c2)])) ? path[(-t0 + c1 - c2) * 2800 + (t0 + c2)] : (path[(-t0 + c1 - c2) * 2800 + c0] + path[c0 * 2800 + (t0 + c2)]));
}
