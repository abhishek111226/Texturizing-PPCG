#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "jacobi-2d_texture_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
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
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_2d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int t, i, j;

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

    
    
    
    cudaArray* cuArr_A;
    cudaChannelFormatDesc channelDesc_A= cudaCreateChannelDesc<double>();
    cudaMallocArray(&cuArr_A, &channelDesc_A, 1300, 1300, cudaArraySurfaceLoadStore);
    cudaMemcpyToArray(cuArr_A, 0, 0, A, (1300) * (1300) * sizeof(double), cudaMemcpyHostToDevice);
    cudaBindSurfaceToArray(surfRef_A, cuArr_A, channelDesc_A);
    
    cudaArray* cuArr_B;
    cudaChannelFormatDesc channelDesc_B= cudaCreateChannelDesc<double>();
    cudaMallocArray(&cuArr_B, &channelDesc_B, 1300, 1300, cudaArraySurfaceLoadStore);
    cudaMemcpyToArray(cuArr_B, 0, 0, B, (1300) * (1300) * sizeof(double), cudaMemcpyHostToDevice);
    cudaBindSurfaceToArray(surfRef_B, cuArr_B, channelDesc_B);
    
    for (int c0 = 0; c0 <= 499; c0 += 1) {
      {
        dim3 k0_dimBlock(1, 1);
        dim3 k0_dimGrid(256, 256);
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (c0);
        cudaCheckKernel();
      }
      
      {
        dim3 k1_dimBlock(1, 1);
        dim3 k1_dimGrid(256, 256);
        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (c0);
        cudaCheckKernel();
      }
      
    }
    cudaMemcpyFromArray(A, cuArr_A, 0, 0, (1300) * (1300) * sizeof(double),  cudaMemcpyDeviceToHost);
    
    cudaMemcpyFromArray(B, cuArr_B, 0, 0, (1300) * (1300) * sizeof(double),  cudaMemcpyDeviceToHost);
    
    
    
    cudaFreeArray(cuArr_A);
    cudaFreeArray(cuArr_B);
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
