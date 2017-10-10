#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void fdtd3d(float*, float*, float*, float*, float*, float*, int, int);
extern "C" void fdtd3d_gold(float*, float*, float*, float*, float*, float*, int, int);

int main(int argc, char** argv) {
  int N = 300;
  int T = 4;

  float (*Ex)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*E_x) = getZero3DArray<float>(N, N, N);
  memcpy (E_x, Ex, sizeof(float)*N*N*N);
  float (*Ey)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*E_y) = getZero3DArray<float>(N, N, N);
  memcpy (E_y, Ey, sizeof(float)*N*N*N);
  float (*Ez)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*E_z) = getZero3DArray<float>(N, N, N);
  memcpy (E_z, Ez, sizeof(float)*N*N*N);
  float (*Hx)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*H_x) = getZero3DArray<float>(N, N, N);
  memcpy (H_x, Hx, sizeof(float)*N*N*N);
  float (*Hy)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*H_y) = getZero3DArray<float>(N, N, N);
  memcpy (H_y, Hy, sizeof(float)*N*N*N);
  float (*Hz)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*H_z) = getZero3DArray<float>(N, N, N);
  memcpy (H_z, Hz, sizeof(float)*N*N*N);

  fdtd3d((float*)Hz, (float*)Hy, (float*)Hx, (float*)Ez, (float*)Ey, (float*)Ex, N, T);
  fdtd3d_gold(H_z, H_y, H_x, E_z, E_y, E_x, N, T);

  double error_z = checkError3D<float> (N, N, (float*)Hz, H_z, 1, N-1, 1, N-1, 1, N-1);
  printf("[Test] RMS Error : %e\n",error_z);
  if (error_z > TOLERANCE)
    return -1;
  double error_y = checkError3D<float> (N, N, (float*)Hy, H_y, 1, N-1, 1, N-1, 1, N-1);
  printf("[Test] RMS Error : %e\n",error_y);
  if (error_y > TOLERANCE)
    return -1;
  double error_x = checkError3D<float> (N, N, (float*)Hx, H_x, 1, N-1, 1, N-1, 1, N-1);
  printf("[Test] RMS Error : %e\n",error_x);
  if (error_x > TOLERANCE)
    return -1;

  delete[] Hx;
  delete[] H_x;
  delete[] Hy;
  delete[] H_y;
  delete[] Hz;
  delete[] H_z;
}
