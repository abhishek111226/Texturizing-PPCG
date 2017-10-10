#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "cholesky_correct_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* cholesky.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "cholesky.h"


/* Array initialization. */
static
void init_array(int n,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j <= i; j++)
	A[i][j] = (DATA_TYPE)(-j % n) / n + 1;
      for (j = i+1; j < n; j++) {
	A[i][j] = 0;
      }
      A[i][i] = 1;
    }

  /* Make the matrix positive semi-definite. */
  int r,s,t;
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (POLYBENCH_ARRAY(B))[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	(POLYBENCH_ARRAY(B))[r][s] += A[r][t] * A[s][t];
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
	A[r][s] = (POLYBENCH_ARRAY(B))[r][s];
  POLYBENCH_FREE_ARRAY(B);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j <= i; j++) {
    if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_cholesky(int n,
		     DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j, k;


  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if (n >= 1) {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

    double *dev_A;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (n) * (2000) * sizeof(double)));
    
    
    cudaCheckReturn(cudaMemcpy(dev_A, A, (n) * (2000) * sizeof(double), cudaMemcpyHostToDevice));
    for (int c0 = 0; c0 < 5997 * n - 5996; c0 += 1) {
      if (n + 1999 * ((5993 * c0 + 3) / 5996) >= 1998 * c0 + 3 && 3997 * ((5993 * c0 + 3) / 5996) >= 3995 * c0 + 3)
        {
          dim3 k0_dimBlock(16, 32);
          dim3 k0_dimGrid(ppcg_min(256, (n + 31) / 32), n >= 2712 && n <= 2722 ? 255 : ppcg_min(256, (3 * n + 25) / 32));
          kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, n, c0);
          cudaCheckKernel();
        }
        
      if ((c0 - 1999) % 5997 == 0)
        {
          dim3 k1_dimBlock(32);
          dim3 k1_dimGrid(ppcg_min(32768, (n + 31) / 32));
          kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, n, c0);
          cudaCheckKernel();
        }
        
      if (5997 * n >= c0 + 7996 && 3998 * ((2 * c0 + 3998) / 5997) >= c0 + 3998 && c0 % 1999 == 0)
        {
          dim3 k2_dimBlock(32);
          dim3 k2_dimGrid(ppcg_min(32768, (n + 31) / 32));
          kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_A, n, c0);
          cudaCheckKernel();
        }
        
      if (c0 % 5997 == 0)
        {
          dim3 k3_dimBlock;
          dim3 k3_dimGrid;
          kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_A, n, c0);
          cudaCheckKernel();
        }
        
    }
    cudaCheckReturn(cudaMemcpy(A, dev_A, (n) * (2000) * sizeof(double), cudaMemcpyDeviceToHost));
    
    
    cudaCheckReturn(cudaFree(dev_A));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_cholesky (n, POLYBENCH_ARRAY(A));

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
