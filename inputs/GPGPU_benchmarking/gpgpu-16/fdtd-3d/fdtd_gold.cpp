#include "../common/common.hpp"

extern "C" void fdtd3d_gold (float *H_z, float *H_y, float *H_x, float *E_z, float *E_y, float *E_x, int N, int T) {
  float (*Hx)[N][N] = (float (*)[N][N])H_x;
  float (*Hy)[N][N] = (float (*)[N][N])H_y;
  float (*Hz)[N][N] = (float (*)[N][N])H_z;
  float (*Ex)[N][N] = (float (*)[N][N])E_x;
  float (*Ey)[N][N] = (float (*)[N][N])E_y;
  float (*Ez)[N][N] = (float (*)[N][N])E_z;

  for (int t=0; t<T; t++) {
    for (int i=1; i<N; i++) {
       for (int j=1; j<N; j++) {
	 for (int k=1; k<N; k++) {
	    Ex[i][j][k] = 0.0002f * Ex[i][j][k] + 0.0005f * (Hz[i][j][k] - Hz[i][j-1][k]) + 0.0005f * (Hy[i][j][k] - Hy[i][j][k-1]);
	    Ey[i][j][k] = 0.0008f * Ey[i][j][k] + 0.0005f * (Hx[i][j][k] - Hx[i][j][k-1]) + 0.0002f * (Hz[i][j][k] - Hz[i-1][j][k]);
	    Ez[i][j][k] = 0.0004f * Ez[i][j][k] + 0.0002f * (Hy[i][j][k] - Hy[i-1][j][k]) + 0.0005f * (Hz[i][j][k] - Hz[i][j-1][k]);
	 }
       }
    }
    for (int i=0; i<N-1; i++) {
      for (int j=0; j<N-1; j++) {
         for (int k=0; k<N-1; k++) {
	    Hx[i][j][k] = Hx[i][j][k] + 0.0003f * (Ez[i][j+1][k] - Ez[i][j][k]) + 0.0007f * (Ey[i][j][k+1] - Ey[i][j][k]);
	    Hy[i][j][k] = Hy[i][j][k] + 0.0004f * (Ex[i][j][k+1] - Ex[i][j][k]) + 0.0008f * (Ez[i+1][j][k] - Ez[i][j][k]);
	    Hz[i][j][k] = Hz[i][j][k] + 0.0005f * (Ey[i+1][j][k] - Ey[i][j][k]) + 0.0009f * (Ez[i][j+1][k] - Ez[i][j][k]);
	 }
      }
    }
  }
}
