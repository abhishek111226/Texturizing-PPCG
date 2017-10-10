#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void fdtd3d(float4*, float4*, int, int, int, float4*);
extern "C" void fdtd3d_gold(float4*, float4*, int, int);

int main(int argc, char** argv) {
  int N = 300;
  int T = 4;

  float4 (*H)[N][N] = (float4 (*)[N][N]) getRandom3DArray<float4>(N,N,N);
  float4 (*E)[N][N] = (float4 (*)[N][N]) getRandom3DArray<float4>(N,N,N);
  float4 (*Op1)[N][N] = (float4 (*)[N][N]) getZero3DArray<float4>(N,N,N);

  fdtd3d ((float4*)H, (float4*)E, N, N, N, (float4*)Op1);
  fdtd3d_gold ((float4*)H, (float4*)E, N, T);

  double error = checkError3D<float4> (N, N, (float4*)Op1, (float4*)H, 1, N-1, 1, N-1, 1, N-1);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] H;
  delete[] E;
  delete[] Op1;
}
