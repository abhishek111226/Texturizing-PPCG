#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "heat2d_gold_kernel.hu"
#include "../common/common.hpp"

static void heat2d_step
(const float* l_input, int width_y, int width_x, float* l_output, int step)
{
  const float (*input)[width_x] = (const float (*)[width_x])l_input;
  float (*output)[width_x] = (float (*)[width_x])l_output;
  for( int i = 1 ; i < width_y-1 ; i++ )
    for( int j = 1 ; j < width_x-1 ; j++ )
      output[i][j] = 0.125f * (input[i+1][j] - 2.0f * input[i][j] + input[i-1][j]) + 
		     0.125f * (input[i][j+1] - 2.0f * input[i][j] + input[i][j-1]) + 
		     input[i][j];
}

//extern "C" 
void heat2d_gold
(float* l_input, int width_y, int width_x, float* l_output)
{
  float* temp = getZero2DArray<float>(width_y, width_x);
  heat2d_step(l_input, width_y, width_x, temp, 0);
  heat2d_step(temp, width_y, width_x, l_output, 1);
  //memset(temp, 0, sizeof(float) * width_y * width_x);
  heat2d_step(l_output, width_y, width_x, temp, 2);
  //memset(l_output, 0, sizeof(float) * width_y * width_x);
  heat2d_step(temp, width_y, width_x, l_output, 3);
}
