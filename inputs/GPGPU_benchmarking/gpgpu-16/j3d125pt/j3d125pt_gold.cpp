#include "../common/common.hpp"


static void j3d_step
(const float* l_in, int height, int width_y, int width_x, float* l_out, int step)
{
  const float (*in)[width_y][width_x] =
    (const float (*)[width_y][width_x])l_in;
  float (*out)[width_y][width_x] = (float (*)[width_y][width_x])l_out;

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width_y-1; j++)
      for (int k = 1; k < width_x-1; k++) {
        out[i][j][k] = (in[i-2][j-1][k-1] + 0.7*in[i-1][j-1][k] + 0.9*in[i-1][j-1][k+1] + 1.2*in[i-1][j][k-1] + 1.5*in[i-1][j][k] + 1.2*in[i-1][j][k+1] + 0.9*in[i-1][j+1][k-1] + 0.7*in[i-1][j+1][k] + 0.5*in[i-1][j+1][k+1] +
                           0.51*in[i][j-1][k-1] + 0.71*in[i][j-1][k] + 0.91*in[i][j-1][k+1] + 1.21*in[i][j][k-1] + 1.51*in[i][j][k] + 1.21*in[i][j][k+1] + 0.91*in[i][j+1][k-1] + 0.71*in[i][j+1][k] + 0.51*in[i][j+1][k+1] +
                           0.52*in[i+1][j-1][k-1] + 0.72*in[i+1][j-1][k] + 0.92*in[i+1][j-1][k+1] + 1.22*in[i+1][j][k-1] + 1.52*in[i+1][j][k] + 1.22*in[i+1][j][k+1] + 0.92*in[i+1][j+1][k-1] + 0.72*in[i+1][j+1][k] + 0.52*in[i+1][j+1][k+1]) / 159;
      }
}


extern "C" void j3d27pt_gold
(float *l_in, int height, int width_y, int width_x, float* l_out)
{
  float* temp = getZero3DArray<float>(height, width_y, width_x);
  j3d_step(l_in, height, width_y, width_x, temp, 0);
  j3d_step(temp, height, width_y, width_x, l_out, 1);
  memset(temp, 0, sizeof(float) * height * width_y * width_x);
  j3d_step(l_out, height, width_y, width_x, temp, 2);
  memset(l_out, 0, sizeof(float) * height * width_y * width_x);
  j3d_step(temp, height, width_y, width_x, l_out, 3);
  delete[] temp;
}
