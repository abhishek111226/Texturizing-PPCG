#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "chol_kernel.hu"
//#include <stdio.h>
int A[3][3];
int B[3][3];
int C[3];
int main()
{
  int i,j,k;
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

    int (*dev_A)[3];
    
    cudaCheckReturn(cudaMalloc((void **) &dev_A, (3) * (3) * sizeof(int)));
    
    
    cudaCheckReturn(cudaMemcpy(dev_A, A, (3) * (3) * sizeof(int), cudaMemcpyHostToDevice));
    for (int c0 = 0; c0 <= 2; c0 += 1) {
      if (c0 >= 1)
        {
          dim3 k0_dimBlock;
          dim3 k0_dimGrid;
          kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, c0);
          cudaCheckKernel();
        }
        
      {
        dim3 k1_dimBlock;
        dim3 k1_dimGrid;
        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_A, c0);
        cudaCheckKernel();
      }
      
      if (c0 <= 1)
        {
          dim3 k2_dimBlock(3);
          dim3 k2_dimGrid(1);
          kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_A, c0);
          cudaCheckKernel();
        }
        
    }
    cudaCheckReturn(cudaMemcpy(A, dev_A, (3) * (3) * sizeof(int), cudaMemcpyDeviceToHost));
    
    
    cudaCheckReturn(cudaFree(dev_A));
  }
printf("hello");
return 0;
}
