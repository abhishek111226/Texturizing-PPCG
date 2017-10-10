#include "../common/common.hpp"

extern "C" void fdtd3d_gold (float4 *h_H, float4 *h_E, int N, int T) {
  float4 (*H)[N][N] = (float4 (*)[N][N])h_H;
  float4 (*E)[N][N] = (float4 (*)[N][N])h_E;

  for (int t=0; t<T; t++) {
    for (int i=1; i<N; i++) {
       for (int j=1; j<N; j++) {
	 for (int k=1; k<N; k++) {
	    E[i][j][k].x = 0.0002f * E[i][j][k].x + 0.0005f * (H[i][j][k].z - H[i][j-1][k].z) + 0.0005f * (H[i][j][k].y - H[i][j][k-1].y);
	    E[i][j][k].y = 0.0008f * E[i][j][k].y + 0.0005f * (H[i][j][k].x - H[i][j][k-1].x) + 0.0002f * (H[i][j][k].z - H[i-1][j][k].z);
	    E[i][j][k].z = 0.0004f * E[i][j][k].z + 0.0002f * (H[i][j][k].y - H[i-1][j][k].y) + 0.0005f * (H[i][j][k].z - H[i][j-1][k].z);
	    E[i][j][k].w = 0.0f;
	 }
       }
    }
    for (int i=0; i<N-1; i++) {
      for (int j=0; j<N-1; j++) {
         for (int k=0; k<N-1; k++) {
	    H[i][j][k].x = H[i][j][k].x + 0.0003f * (E[i][j+1][k].z - E[i][j][k].z) + 0.0007f * (E[i][j][k+1].y - E[i][j][k].y);
	    H[i][j][k].y = H[i][j][k].y + 0.0004f * (E[i][j][k+1].x - E[i][j][k].x) + 0.0008f * (E[i+1][j][k].z - E[i][j][k].z);
	    H[i][j][k].z = H[i][j][k].z + 0.0005f * (E[i+1][j][k].y - E[i][j][k].y) + 0.0009f * (E[i][j+1][k].z - E[i][j][k].z);
	    H[i][j][k].w = 0.0f;
	 }
      }
    }
  }
}
