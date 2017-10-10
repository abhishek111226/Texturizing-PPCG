#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "j3d27pt_kernel.hu"
/**
 * jacobi-2d-imper.c: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
extern "C"{
#include <polybench.h>
}
/* Include benchmark-specific header. */
/* Default data type is double, default size is 20x1000. */
#include "jacobi-3d-imper.h"


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

	   float (*dev_B)[1000][1000];
	   
	   cudaCheckReturn(cudaMalloc((void **) &dev_B, (999) * (1000) * (1000) * sizeof(float)));
	   
	   
	   cudaArray* cuArr_A;
	   cudaChannelFormatDesc channelDesc_A= cudaCreateChannelDesc<float>();
	   const cudaExtent extent_A= make_cudaExtent(1000, 1000, 1000);
	   cudaMalloc3DArray(&cuArr_A, &channelDesc_A, extent_A);
	   cudaMemcpy3DParms copyParams_A_to_device= {0};
	   copyParams_A_to_device.srcPtr = make_cudaPitchedPtr((void*)A, extent_A.width*sizeof(float), extent_A.width, extent_A.height);
	   copyParams_A_to_device.dstArray = cuArr_A;
	   copyParams_A_to_device.extent = extent_A;
	   copyParams_A_to_device.kind = cudaMemcpyHostToDevice;
	   cudaMemcpy3D(&copyParams_A_to_device);
	   cudaBindTextureToArray(texRef_A, cuArr_A, channelDesc_A);
	   
	   cudaCheckReturn(cudaMemcpy(dev_B, B, (999) * (1000) * (1000) * sizeof(float), cudaMemcpyHostToDevice));
	   {
	     dim3 k0_dimBlock(4, 4, 32);
	     dim3 k0_dimGrid(32, 32);
	     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_B);
	     cudaCheckKernel();
	   }
	   
	   cudaCheckReturn(cudaMemcpy(B, dev_B, (999) * (1000) * (1000) * sizeof(float), cudaMemcpyDeviceToHost));
	   
	   cudaUnbindTexture(texRef_A);
	   
	   cudaFreeArray(cuArr_A);
	   cudaCheckReturn(cudaFree(dev_B));
	 }

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
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
