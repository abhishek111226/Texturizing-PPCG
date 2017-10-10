#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "gramschmidt_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gramschmidt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gramschmidt.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((DATA_TYPE) ((i*j) % m) / m )*100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("R");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, R[i][j]);
    }
  POLYBENCH_DUMP_END("R");

  POLYBENCH_DUMP_BEGIN("Q");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i*n+j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, Q[i][j]);
    }
  POLYBENCH_DUMP_END("Q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* QR Decomposition with Modified Gram Schmidt:
 http://www.inf.ethz.ch/personal/gander/ */
static
void kernel_gramschmidt(int m, int n,
			DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
			DATA_TYPE POLYBENCH_2D(R,N,N,n,n),
			DATA_TYPE POLYBENCH_2D(Q,M,N,m,n))
{
  int i, j, k;

  DATA_TYPE nrm;

  {
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

    float (*dev_A)[1200];
    float (*dev_Q)[1200];
    float (*dev_R)[1200];
    float *dev_nrm;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (1000) * (1200) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_Q, (1000) * (1200) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_R, (1200) * (1200) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_nrm, sizeof(float)));
    
    
    cudaCheckReturn(cudaMemcpy(dev_A, A, (1000) * (1200) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_R, R, (1200) * (1200) * sizeof(float), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(16, 32);
      dim3 k0_dimGrid(38, 38);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_R);
      cudaCheckKernel();
    }
    
    for (int c0 = 0; c0 <= 7191602; c0 += 1) {
      if (c0 <= 7185607 && c0 % 5993 == 0)
        {
          dim3 k1_dimBlock;
          dim3 k1_dimGrid;
          kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_nrm, c0);
          cudaCheckKernel();
        }
        
      if ((c0 - 3597) % 5995 == 0) {
        {
          dim3 k2_dimBlock(32);
          dim3 k2_dimGrid(32);
          kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_A, dev_Q, dev_R, c0);
          cudaCheckKernel();
        }
        
      } else if ((c0 - 1199) % 5995 == 0)
        {
          dim3 k3_dimBlock;
          dim3 k3_dimGrid;
          kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_A, dev_nrm, c0);
          cudaCheckKernel();
        }
        
      if (5995 * ((c0 + 1199) / 2398) + 3591005 >= 3 * c0 && 5995 * ((2 * c0 + 2398) / 3597) >= 3 * c0 + 5995 && c0 % 1199 == 0)
        {
          dim3 k4_dimBlock(16, 32);
          dim3 k4_dimGrid(32, 75);
          kernel4 <<<k4_dimGrid, k4_dimBlock>>> (dev_A, dev_Q, dev_R, c0);
          cudaCheckKernel();
        }
        
      if (5995 * (c0 / 2398) + 3593403 >= 3 * c0 && 5995 * ((2 * c0 + 1199) / 3597) >= 3 * c0 + 3597 && c0 % 1199 == 0)
        {
          dim3 k5_dimBlock(32);
          dim3 k5_dimGrid(75);
          kernel5 <<<k5_dimGrid, k5_dimBlock>>> (dev_A, dev_Q, dev_R, c0);
          cudaCheckKernel();
        }
        
      if ((c0 - 2398) % 5995 == 0)
        {
          dim3 k6_dimBlock;
          dim3 k6_dimGrid;
          kernel6 <<<k6_dimGrid, k6_dimBlock>>> (dev_R, dev_nrm, c0);
          cudaCheckKernel();
        }
        
    }
    cudaCheckReturn(cudaMemcpy(A, dev_A, (1000) * (1200) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(Q, dev_Q, (1000) * (1200) * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheckReturn(cudaMemcpy(R, dev_R, (1200) * (1200) * sizeof(float), cudaMemcpyDeviceToHost));
    
    
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_Q));
    cudaCheckReturn(cudaFree(dev_R));
    cudaCheckReturn(cudaFree(dev_nrm));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,N,N,n,n);
  POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(R),
	      POLYBENCH_ARRAY(Q));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_gramschmidt (m, n,
		      POLYBENCH_ARRAY(A),
		      POLYBENCH_ARRAY(R),
		      POLYBENCH_ARRAY(Q));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(R);
  POLYBENCH_FREE_ARRAY(Q);

  return 0;
}
