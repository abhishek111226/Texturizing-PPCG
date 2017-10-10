#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "if_else_kernel.hu"
int tex_array[10][10];
//int cangle[360];
int main()
{	
int t, i, j,loop,sum1,sum2,a;
 for(int j=0;j<10;j++)
 for(int i=0;i<10;i++)
 {

	tex_array[i][j]=i+j;
 }
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

   int *dev_tex_array;
   
   cudaCheckReturn(cudaMalloc((void **) &dev_tex_array, (10) * (10) * sizeof(int)));
   
   
   cudaBindTexture(NULL,texRef_tex_array, dev_tex_array, (10) * (10) * sizeof(int));
   
   cudaCheckReturn(cudaMemcpy(dev_tex_array, tex_array, (10) * (10) * sizeof(int), cudaMemcpyHostToDevice));
   {
     dim3 k0_dimBlock(10, 10);
     dim3 k0_dimGrid(1, 1);
     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_tex_array);
     cudaCheckKernel();
   }
   
   cudaCheckReturn(cudaMemcpy(tex_array, dev_tex_array, (10) * (10) * sizeof(int), cudaMemcpyDeviceToHost));
   
   cudaUnbindTexture(texRef_tex_array);
   
   cudaCheckReturn(cudaFree(dev_tex_array));
 }
 for(int j=0;j<10;j++)
 for(int i=0;i<10;i++)
 {
	sum1+=tex_array[i][j];
 }
  return 0;
}
