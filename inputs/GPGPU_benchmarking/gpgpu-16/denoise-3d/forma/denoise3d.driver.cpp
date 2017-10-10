#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void denoise3d (float *, float *, int);
extern "C" void denoise3d_gold (float *, float *, int);

int main(int argc, char** argv) {
  int N = 500;

  float (*u0)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*f)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*u1)[N][N] = (float (*)[N][N]) getZero3DArray<float>(N, N, N);
  memcpy (u1, u0, sizeof(float)*N*N*N);

  denoise3d((float*)u1, (float*)f, N);
  denoise3d_gold((float*)u0, (float*)f, N);

  double error = checkError3D<float> (N, N, (float*)u0, (float*)u1, 8, N-8, 8, N-8, 8, N-8);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] u0;
  delete[] u1;
  delete[] f;
}
