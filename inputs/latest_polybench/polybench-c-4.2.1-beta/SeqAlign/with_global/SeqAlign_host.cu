#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "SeqAlign_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
extern "C"{
#include <polybench.h>
}
/* Include benchmark-specific header. */
#include "mvt.h"


/* Array initialization. */
static
void init_array(int q, int d,
		DATA_TYPE POLYBENCH_1D(QS,Q,q),
		DATA_TYPE POLYBENCH_1D(DB,D,d),
		DATA_TYPE POLYBENCH_2D(MatchQ,Q,D,q,d))
{
  int i, j;
  for(i=0;i< q; i++)
	QS[i] = rand() % ('Z' - 'A' + 1);
  for(j=0;j< d; j++)
	DB[j] = rand() % ('Z' - 'A' + 1);

  for(i=0;i< q; i++)
	for(j=0;j< d; j++)
	{
		if(QS[i] == DB[j])
			MatchQ[i][j] =	15; 
		else 
			MatchQ[i][j] = -12;
	}

}

DATA_TYPE max4(DATA_TYPE a, DATA_TYPE b, DATA_TYPE c, DATA_TYPE d)
{
    DATA_TYPE max;	
    if ( a > b ) {
        max = a;
    } else {
        max = b;
    }
    
    if ( c > max ) {
        max = c;
    }
    
    if ( d > max ) {
        max = d;
}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int q, int d,
		DATA_TYPE POLYBENCH_1D(QS,Q,q),
		DATA_TYPE POLYBENCH_1D(DB,D,d),
		DATA_TYPE POLYBENCH_2D(MatchQ,Q,D,q,d))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("QS");
  for (i = 0; i < q; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, QS[i]);
  }
  POLYBENCH_DUMP_END("QS");
  POLYBENCH_DUMP_BEGIN("DB");
  for (i = 0; i < d; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, DB[i]);
  }
  POLYBENCH_DUMP_END("DB");
  POLYBENCH_DUMP_BEGIN("MatchQ");
  for (i = 0; i < q; i++)
    for (j = 0; j < d; j++) {
	if ((i * q + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, MatchQ[i][j]);
    }
  POLYBENCH_DUMP_END("MatchQ");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_mvt(int q, int d,
		DATA_TYPE POLYBENCH_1D(QS,Q,q),
		DATA_TYPE POLYBENCH_1D(DB,D,d),
		DATA_TYPE POLYBENCH_2D(MatchQ,Q,D,q,d),
		DATA_TYPE POLYBENCH_2D(M,Q,D,q,d))
{
  int i, j;

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

  float (*dev_M)[2000];
  float (*dev_MatchQ)[2000];
  
  cudaCheckReturn(cudaMalloc((void **) &dev_M, (3000) * (2000) * sizeof(float)));
  cudaCheckReturn(cudaMalloc((void **) &dev_MatchQ, (3000) * (2000) * sizeof(float)));
  
  
  cudaCheckReturn(cudaMemcpy(dev_MatchQ, MatchQ, (3000) * (2000) * sizeof(float), cudaMemcpyHostToDevice));
  {
    dim3 k0_dimBlock(16, 32);
    dim3 k0_dimGrid(63, 94);
    kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_M);
    cudaCheckKernel();
  }
  
  for (int c0 = 2; c0 <= 4998; c0 += 1)
    {
      dim3 k1_dimBlock(32);
      dim3 k1_dimGrid(63);
      kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_M, dev_MatchQ, c0);
      cudaCheckKernel();
    }
    
  cudaCheckReturn(cudaMemcpy(M, dev_M, (3000) * (2000) * sizeof(float), cudaMemcpyDeviceToHost));
  
  
  cudaCheckReturn(cudaFree(dev_M));
  cudaCheckReturn(cudaFree(dev_MatchQ));
}

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int q = Q;
  int d = D;	

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(M, DATA_TYPE, Q, D, q, d);
  POLYBENCH_2D_ARRAY_DECL(MatchQ, DATA_TYPE, Q, D, q, d);
  POLYBENCH_1D_ARRAY_DECL(QS, DATA_TYPE, Q, q);
  POLYBENCH_1D_ARRAY_DECL(DB, DATA_TYPE, D, d);


  /* Initialize array(s). */
 init_array (q, d,
	      POLYBENCH_ARRAY(QS),
	      POLYBENCH_ARRAY(DB),
	      POLYBENCH_ARRAY(MatchQ));

  /* Start timer. */
  polybench_start_instruments;

/*
DATA_TYPE POLYBENCH_1D(QS,Q,q),
		DATA_TYPE POLYBENCH_1D(DB,D,d),
		DATA_TYPE POLYBENCH_2D(MatchQ,Q,D,q,d),
		DATA_TYPE POLYBENCH_2D(M,Q,D,q,d))
*/

  /* Run kernel. */
  kernel_mvt (q, d,
	      POLYBENCH_ARRAY(QS),
	      POLYBENCH_ARRAY(DB),
	      POLYBENCH_ARRAY(MatchQ),
	      POLYBENCH_ARRAY(M));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(q, d, 
	      POLYBENCH_ARRAY(QS),
	      POLYBENCH_ARRAY(DB),
	      POLYBENCH_ARRAY(MatchQ)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(M);
  POLYBENCH_FREE_ARRAY(QS);
  POLYBENCH_FREE_ARRAY(DB);
  POLYBENCH_FREE_ARRAY(MatchQ);

  return 0;
}
