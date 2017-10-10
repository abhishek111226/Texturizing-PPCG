#include "../common/common.hpp"

static void minigmg_step
(float a, float b, float h2inv, const float* l_alpha, const float* l_input, const float* l_beta_i, const float* l_beta_j, const float* l_beta_k, const float* l_lambda, const float* l_rhs, int L, int M, int N, float* l_output) {
  const float (*alpha)[M][N] = (const float(*)[M][N])l_alpha;
  const float (*input)[M][N] = (const float(*)[M][N])l_input;
  const float (*beta_i)[M][N] = (const float(*)[M][N])l_beta_i;
  const float (*beta_j)[M][N] = (const float(*)[M][N])l_beta_j;
  const float (*beta_k)[M][N] = (const float(*)[M][N])l_beta_k;
  const float (*lambda)[M][N] = (const float(*)[M][N])l_lambda;
  const float (*rhs)[M][N] = (const float(*)[M][N])l_rhs;
  float (*output)[M][N] = (float(*)[M][N])l_output;
  for (int i = 1; i < L-1; i++)
    for (int j = 1; j < M-1; j++)
      for (int k = 1; k < N-1; k++) {
        float helmholtz = a * alpha[i][j][k] * input[i][j][k] -
          b * h2inv * (beta_i[i][j][k+1] * (input[i][j][k+1] - input[i][j][k])
            		   - beta_i[i][j][k] * (input[i][j][k] - input[i][j][k-1])
            		   + beta_j[i][j+1][k] * (input[i][j+1][k] - input[i][j][k])
            		   - beta_j[i][j][k] * (input[i][j][k] - input[i][j-1][k])
            		   + beta_k[i+1][j][k] * (input[i+1][j][k] - input[i][j][k])
            		   - beta_k[i][j][k] * (input[i][j][k] - input[i-1][j][k]));
        output[i][j][k] = input[i][j][k] - 2.0f * lambda[i][j][k] * (helmholtz - rhs[i][j][k]) / 3.0f;
      }
}


extern "C" void minigmg_gold (float a, float b, float h2inv, const float *alpha, const float* phi, const float* beta_i, const float* beta_j, const float* beta_k, const float* lambda, const float* rhs, int L, int M, int N, float* output_gold) {
  float *temp = getZero3DArray<float>(L,M,N);
  minigmg_step (a, b, h2inv, alpha, phi, beta_i, beta_j, beta_k, lambda, rhs, L, M, N, temp);
  memset(output_gold, 0, sizeof(float)*L*M*N);
  minigmg_step (a, b, h2inv, alpha, temp, beta_i, beta_j, beta_k, lambda, rhs, L, M, N, output_gold);
  memset(temp, 0, sizeof(float)*L*M*N);
  minigmg_step (a, b, h2inv, alpha, output_gold, beta_i, beta_j, beta_k, lambda, rhs, L, M, N, temp);
  memcpy(output_gold, temp, sizeof(float)*L*M*N);
  minigmg_step (a, b, h2inv, alpha, temp, beta_i, beta_j, beta_k, lambda, rhs, L, M, N, output_gold);
  delete[] temp;
}
