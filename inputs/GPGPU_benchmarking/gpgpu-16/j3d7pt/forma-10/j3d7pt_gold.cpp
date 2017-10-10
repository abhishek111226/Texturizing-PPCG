#include "../../common/common.hpp"


static void j3d_step
(const float* l_input, int height, int width_y, int width_x, float* l_output, int step)
{
  const float (*input)[width_y][width_x] =
    (const float (*)[width_y][width_x])l_input;
  float (*output)[width_y][width_x] = (float (*)[width_y][width_x])l_output;

  for (int i = 1; i < height-1; i++)
    for (int j = 1; j < width_y-1; j++)
      for (int k = 1; k < width_x-1; k++) {
        output[i][j][k] =
          (0.161f * input[i][j][k+1] + 0.162f * input[i][j][k-1] +
          0.163f * input[i][j+1][k] + 0.164f * input[i][j-1][k] +
          0.165f * input[i+1][j][k] + 0.166f * input[i-1][j][k] -
          1.67f * input[i][j][k])/2;
	  //if (step == 0) printf ("output[%d][%d][%d] = %.6f (%.6f)\n", i, j, k, output[i][j][k], input[i+1][j][k]);
      }
}


extern "C" void j3d7pt_gold
(float *l_input, int height, int width_y, int width_x, float* l_output)
{
  float* temp = getZero3DArray<float>(height, width_y, width_x);
  memcpy (temp, l_input, sizeof(float)*height*width_x*width_y);
  for (int t=0; t<10; t++) {
    j3d_step(temp, height, width_y, width_x, l_output, t);
    memcpy (temp, l_output, sizeof(float)*height*width_x*width_y);
  }
  delete[] temp;
}
