/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t, i, j, k;
#pragma scop
#pragma texture (A, B)
    for (t = 1; t <= TSTEPS; t++) {
       for (int i = 1; i < _PB_N-1; i++)
    		for (int j = 1; j < _PB_N-1; j++)
      			for (int k = 1; k < _PB_N-1; k++) {
		         B[i][j][k] = (0.5*A[i-1][j-1][k-1] + 0.7*A[i-1][j-1][k] + 0.9*A[i-1][j-1][k+1] + 						    1.2*A[i-1][j][k-1] + 1.5*A[i-1][j][k] + 1.2*A[i-1][j][k+1] + 
					    0.9*A[i-1][j+1][k-1] + 0.7*A[i-1][j+1][k] + 0.5*A[i-1][j+1][k+1] +
                                            0.51*A[i][j-1][k-1] + 0.71*A[i][j-1][k] +   0.91*A[i][j-1][k+1] + 						    1.21*A[i][j][k-1] + 1.51*A[i][j][k] + 1.21*A[i][j][k+1] + 
					    0.91*A[i][j+1][k-1] + 0.71*A[i][j+1][k] + 0.51*A[i][j+1][k+1] +
                           		    0.52*A[i+1][j-1][k-1] + 0.72*A[i+1][j-1][k] + 0.92*A[i+1][j-1][k+1] + 						    1.22*A[i+1][j][k-1] + 1.52*A[i+1][j][k] + 1.22*A[i+1][j][k+1] + 					    	    0.92*A[i+1][j+1][k-1] + 0.72*A[i+1][j+1][k] + 0.52*A[i+1][j+1][k+1]) / 159;
      }
      for (int i = 1; i < _PB_N-1; i++)
    		for (int j = 1; j < _PB_N-1; j++)
      			for (int k = 1; k < _PB_N-1; k++) {
		         A[i][j][k] = (0.5*B[i-1][j-1][k-1] + 0.7*B[i-1][j-1][k] + 0.9*B[i-1][j-1][k+1] + 						    1.2*B[i-1][j][k-1] + 1.5*B[i-1][j][k] + 1.2*B[i-1][j][k+1] + 
					    0.9*B[i-1][j+1][k-1] + 0.7*B[i-1][j+1][k] + 0.5*B[i-1][j+1][k+1] +
                                            0.51*B[i][j-1][k-1] + 0.71*B[i][j-1][k] +   0.91*B[i][j-1][k+1] + 						    1.21*B[i][j][k-1] + 1.51*B[i][j][k] + 1.21*B[i][j][k+1] + 
					    0.91*B[i][j+1][k-1] + 0.71*B[i][j+1][k] + 0.51*B[i][j+1][k+1] +
                           		    0.52*B[i+1][j-1][k-1] + 0.72*B[i+1][j-1][k] + 0.92*B[i+1][j-1][k+1] + 						    1.22*B[i+1][j][k-1] + 1.52*B[i+1][j][k] + 1.22*B[i+1][j][k+1] + 					    0.92*B[i+1][j+1][k-1] + 0.72*B[i+1][j+1][k] + 
					    0.52*B[i+1][j+1][k+1]) / 159;

      }
    }
#pragma endscop


}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_heat_3d (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

  return 0;
}
