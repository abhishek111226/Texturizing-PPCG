#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "lennardJones_kernel.hu"
/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lennardJones.h"
/* Default data type is double, default size is N=1024. */
//#include "template-for-new-benchmark.h"


/* Array initialization. */
static
void init_array(int n, DATA_TYPE POLYBENCH_2D(x,N,3,n,3), DATA_TYPE POLYBENCH_2D(f,N,3,n,3) )
{
  int i, j;
  double rho = 0.8442;
  double delta = 0.2662;	
  double alat = pow((4.0 / rho), (1.0 / 3.0));	
  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++)
    {
	  x[i][j] =  0.5 * alat * j + delta/1000*(rand()%1000-500);
	  f[i][j] = 0;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, DATA_TYPE POLYBENCH_2D(f,N,3,n,3))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, f[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_template(int n, DATA_TYPE POLYBENCH_2D(x,N,3,n,3), DATA_TYPE POLYBENCH_2D(f,N,3,n,3))
{
  printf(" Running on %d",n);
  int i, j;
  double cutforcesq=FORCE_CUTSEQ;            //force cutoff
  double epsilon=1.0;               //Potential parameter
  double sigma6=1.0; //Potential parameter
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

    float (*dev_f)[3];
    float (*dev_x)[3];
    
    cudaCheckReturn(cudaMalloc((void **) &dev_f, (8400) * (3) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_x, (8400) * (3) * sizeof(float)));
    
    
    cudaCheckReturn(cudaMemcpy(dev_f, f, (8400) * (3) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckReturn(cudaMemcpy(dev_x, x, (8400) * (3) * sizeof(float), cudaMemcpyHostToDevice));
    {
      dim3 k0_dimBlock(32);
      dim3 k0_dimGrid(263);
      kernel0 <<<k0_dimGrid, k0_dimBlock>>> (cutforcesq, epsilon, dev_f, sigma6, dev_x);
      cudaCheckKernel();
    }
    
    cudaCheckReturn(cudaMemcpy(f, dev_f, (8400) * (3) * sizeof(float), cudaMemcpyDeviceToHost));
    
    
    cudaCheckReturn(cudaFree(dev_f));
    cudaCheckReturn(cudaFree(dev_x));
  }

}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(x,DATA_TYPE,N,3,n,3);
  POLYBENCH_2D_ARRAY_DECL(f,DATA_TYPE,N,3,n,3);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(f));


  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_template (n, POLYBENCH_ARRAY(x),POLYBENCH_ARRAY(f));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n,  POLYBENCH_ARRAY(f)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(f);
  POLYBENCH_FREE_ARRAY(x);

  return 0;
}
