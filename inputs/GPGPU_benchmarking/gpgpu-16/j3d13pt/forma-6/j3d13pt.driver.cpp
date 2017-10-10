#include "../../common/common.hpp"
#include "../../common/timer.hpp"
#include <cassert>
#include <cstdio>

extern "C" void j3d13pt(float*, int, int, int, float*);

extern "C" void j3d13pt_gold(float*, int, int, int, float*);

int main(int argc, char** argv) {
  int height, width_y, width_x;
  if (argc == 4) {
    height = atoi(argv[1]);
    width_y = atoi(argv[2]);
    width_x = atoi(argv[3]);
  }
  else {
    height = 512;
    width_y = 512;
    width_x = 512;
  }

  float (*input)[width_y][width_x] = (float (*)[width_y][width_x])
    getRandom3DArray<float>(height, width_y, width_x);
  float (*output)[width_y][width_x] = (float (*)[width_y][width_x])
    getZero3DArray<float>(height, width_y, width_x);
  float (*output_gold)[width_y][width_x] = (float (*)[width_y][width_x])
    getZero3DArray<float>(height, width_y, width_x);

  j3d13pt((float*)input, height, width_y, width_x, (float*)output);

  j3d13pt_gold((float*)input, height, width_y, width_x, (float*)output_gold);

  double error = checkError3D<float> (width_y, width_x, (float*)output, (float*) output_gold, 12, height-12, 12, width_y-12, 12, width_x-12);
  printf("[Test] RMS Error : %e\n",error);
  if (error > TOLERANCE)
    return -1;

  double time = 0.0;
  for (int i = 0; i < NUMRUNS; i++) {
    memset((void*)output, 0, sizeof(float)*height*width_y*width_x);
    startTimer();
    j3d13pt((float*)input, height, width_y, width_x, (float*)output);
    stopTimer();
    time += getElapsedTime();
  }
  printf
    ("[Test] Elapsed Time (average of %d runs) : %e (ms)\n", NUMRUNS,
     time/NUMRUNS);

  delete[] input;
  delete[] output;
  delete[] output_gold;
}
