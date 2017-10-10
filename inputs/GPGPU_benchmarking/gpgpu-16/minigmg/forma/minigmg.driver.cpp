#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void minigmg_gold
(float a, float b, float h2inv, const float *alpha, const float* phi,
 const float* beta_i, const float* beta_j, const float* beta_k,
 const float* lambda, const float* rhs, int L, int M, int N,
 float* output_gold);

extern "C" void minigmg
(float a, float b, float h2inv, float *alpha, float* phi, float* beta_i,
 float* beta_j, float* beta_k, float* lambda, float* rhs, int L, int M, int N,
 float* output);


int main(int argc, char** argv)
{
  int N = 480;

  float a = get_random<float>();
  float b = get_random<float>();
  float h2inv = get_random<float>();

  float (*alpha)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*phi)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*beta_i)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*beta_j)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*beta_k)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*lambda)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N);
  float (*rhs)[N][N] = (float (*)[N][N]) getRandom3DArray<float>(N, N, N); 

  float (*output)[N][N] = (float (*)[N][N]) getZero3DArray<float>(N, N, N);
  float (*output_gold)[N][N] = (float (*)[N][N]) getZero3DArray<float>(N, N, N);

  minigmg (a, b, h2inv, (float*)alpha, (float*)phi, (float*)beta_i, (float*)beta_j, (float*)beta_k, (float*)lambda, (float*)rhs, N, N, N, (float*)output);

  minigmg_gold (a, b, h2inv, (float*)alpha, (float*)phi, (float*)beta_i, (float*)beta_j, (float*)beta_k, (float*)lambda, (float*)rhs, N, N, N, (float*)output_gold);

  double error = checkError3D<float> (N, N, (float*)output, (float*)output_gold, 4 , N - 4, 4, N - 4, 4, N - 4);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] alpha;
  delete[] phi;
  delete[] beta_i;
  delete[] beta_j;
  delete[] beta_k;
  delete[] lambda;
  delete[] rhs;
  delete[] output;
  delete[] output_gold;
}
