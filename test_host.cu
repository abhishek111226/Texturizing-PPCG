#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "test_kernel.hu"
//#include <stdio.h>
int A[4][4];
int B[4][4];
int C[4][4];
int main()
{
 int i,j=0,k; 	
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

   int *dev_A;
   int *dev_B;
   
   cudaCheckReturn(cudaMalloc((void **) &dev_A, (4) * (4) * sizeof(int)));
   cudaCheckReturn(cudaMalloc((void **) &dev_B, (4) * (4) * sizeof(int)));
   
   
   cudaBindTexture(NULL,texRef_A, dev_A, (4) * (4) * sizeof(int));
   
   cudaBindTexture(NULL,texRef_B, dev_B, (4) * (4) * sizeof(int));
   
   cudaCheckReturn(cudaMemcpy(dev_A, A, (4) * (4) * sizeof(int), cudaMemcpyHostToDevice));
   cudaCheckReturn(cudaMemcpy(dev_B, B, (4) * (4) * sizeof(int), cudaMemcpyHostToDevice));
   {
     dim3 k0_dimBlock(4);
     dim3 k0_dimGrid(1);
     kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_B);
     cudaCheckKernel();
   }
   
   cudaCheckReturn(cudaMemcpy(B, dev_B, (4) * (4) * sizeof(int), cudaMemcpyDeviceToHost));
   
   cudaUnbindTexture(texRef_A);
   cudaUnbindTexture(texRef_B);
   
   cudaCheckReturn(cudaFree(dev_A));
   cudaCheckReturn(cudaFree(dev_B));
 }
  return 0;
}

