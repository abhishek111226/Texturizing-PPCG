#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "doitgen_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* doitgen.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
extern "C"{
#include <polybench.h>
}
/* Include benchmark-specific header. */
#include "doitgen.h"


/* Array initialization. */
static
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
  int i, j, k;

  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++)
	A[i][j][k] = (DATA_TYPE) ((i*j + k)%np) / np;
  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++)
      C4[i][j] = (DATA_TYPE) (i*j % np) / np;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < nr; i++)
    for (j = 0; j < nq; j++)
      for (k = 0; k < np; k++) {
	if ((i*nq*np+j*np+k) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgen(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_1D(sum,NP,np))
{
  int r, q, p, s;

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

    float *dev_sum;
    
    cudaCheckReturn(cudaMalloc((void **) &dev_sum, (160) * sizeof(float)));
    
    
    cudaArray* cuArr_A;
    cudaChannelFormatDesc channelDesc_A= cudaCreateChannelDesc<float>();
    const cudaExtent extent_A= make_cudaExtent(160, 140, 150);
    cudaMalloc3DArray(&cuArr_A, &channelDesc_A, extent_A, cudaArraySurfaceLoadStore);
    cudaMemcpy3DParms copyParams_A_to_device= {0};
    copyParams_A_to_device.srcPtr = make_cudaPitchedPtr((void*)A, extent_A.width*sizeof(float), extent_A.width, extent_A.height);
    copyParams_A_to_device.dstArray = cuArr_A;
    copyParams_A_to_device.extent = extent_A;
    copyParams_A_to_device.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams_A_to_device);
    cudaBindSurfaceToArray(surfRef_A, cuArr_A, channelDesc_A);
    
    cudaArray* cuArr_C4;
    cudaChannelFormatDesc channelDesc_C4= cudaCreateChannelDesc<float>();
    cudaMallocArray(&cuArr_C4, &channelDesc_C4, 160, 160);
    cudaMemcpyToArray(cuArr_C4, 0, 0, C4, (160) * (160) * sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTextureToArray(texRef_C4, cuArr_C4, channelDesc_C4);
    
    for (int c0 = 0; c0 <= 149; c0 += 1)
      for (int c1 = 0; c1 <= 139; c1 += 1) {
        {
          dim3 k0_dimBlock(32);
          dim3 k0_dimGrid(5);
          kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_sum, c0, c1);
          cudaCheckKernel();
        }
        
        {
          dim3 k1_dimBlock(32);
          dim3 k1_dimGrid(5);
          kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_sum, c0, c1);
          cudaCheckKernel();
        }
        
        {
          dim3 k2_dimBlock(32);
          dim3 k2_dimGrid(5);
          kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_sum, c0, c1);
          cudaCheckKernel();
        }
        
      }
    cudaMemcpy3DParms copyParams_A_from_device= {0};
    copyParams_A_from_device.dstPtr = make_cudaPitchedPtr((void*)A, extent_A.width*sizeof(float), extent_A.width, extent_A.height);
    copyParams_A_from_device.srcArray = cuArr_A;
    copyParams_A_from_device.extent = extent_A;
    copyParams_A_from_device.kind = cudaMemcpyDeviceToHost;
    cudaMemcpy3D(&copyParams_A_from_device);
    
    cudaCheckReturn(cudaMemcpy(sum, dev_sum, (160) * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaUnbindTexture(texRef_C4);
    
    cudaFreeArray(cuArr_A);
    cudaFreeArray(cuArr_C4);
    cudaCheckReturn(cudaFree(dev_sum));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int nr = NR;
  int nq = NQ;
  int np = NP;

  /* Variable declaration/allocation. */
  POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
  POLYBENCH_1D_ARRAY_DECL(sum,DATA_TYPE,NP,np);
  POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

  /* Initialize array(s). */
  init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_doitgen (nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nr, nq, np,  POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(sum);
  POLYBENCH_FREE_ARRAY(C4);

  return 0;
}
