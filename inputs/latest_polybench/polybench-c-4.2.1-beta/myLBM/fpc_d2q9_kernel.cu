#include <stdio.h> 
#define DEVICECODE true 
#include "fpc_d2q9_kernel.hu"
__global__ void kernel0(double grid[2][262][1028][9], int _nTimesteps, int _nY, int _nX, int c0)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
    for (int c1 = 32 * b0; c1 < _nY; c1 += 8192)
      if (t0 + c1 >= 3 && _nY >= t0 + c1 + 1)
        for (int c2 = 32 * b1; c2 < _nX; c2 += 8192)
          for (int c4 = ppcg_max(t1, ((t1 + c2 + 14) % 16) - c2 + 2); c4 <= ppcg_min(31, _nX - c2 - 1); c4 += 16)
            lbm_kernel(grid[c0 % 2][t0 + c1][c2 + c4][0], grid[c0 % 2][t0 + c1 - 1][c2 + c4][3], grid[c0 % 2][t0 + c1 + 1][c2 + c4][4], grid[c0 % 2][t0 + c1][c2 + c4 - 1][1], grid[c0 % 2][t0 + c1][c2 + c4 + 1][2], grid[c0 % 2][t0 + c1 - 1][c2 + c4 - 1][5], grid[c0 % 2][t0 + c1 - 1][c2 + c4 + 1][6], grid[c0 % 2][t0 + c1 + 1][c2 + c4 - 1][7], grid[c0 % 2][t0 + c1 + 1][c2 + c4 + 1][8], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][0], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][3], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][4], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][1], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][2], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][5], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][6], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][7], &grid[(c0 + 1) % 2][t0 + c1][c2 + c4][8], (c0), (t0 + c1), (c2 + c4));
}
