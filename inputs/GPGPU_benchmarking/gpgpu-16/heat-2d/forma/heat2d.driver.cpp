#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void heat2d(float*, int, int, float*);
extern "C" void heat2d_gold(float*, int, int, float*);

int main(int argc, char** argv) {
  int width_y, width_x;
  if (argc == 3) {
    width_y = atoi(argv[1]);
    width_x = atoi(argv[2]);
  }
  else {
    width_y = 8192;
    width_x = 8192;
  }

  float (*input)[width_x] = (float (*)[width_x]) getRandom2DArray<float>(width_y, width_x);
  float (*output)[width_x] = (float (*)[width_x]) getZero2DArray<float>(width_y, width_x);
  float (*output_gold)[width_x] = (float (*)[width_x]) getZero2DArray<float>(width_y, width_x);

  heat2d((float*)input, width_y, width_x, (float*)output);
  heat2d_gold((float*)input, width_y, width_x, (float*)output_gold);

  double error = checkError2D<float> (width_x, (float*)output, (float*) output_gold, 4, width_y-4, 4, width_x-4);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
