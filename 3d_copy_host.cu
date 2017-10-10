#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "3d_copy_kernel.hu"
//#include <stdio.h>
int A[300][300][300];
int B[300][300][300];
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

   
   
   
   cudaArray* cuArr_A;
   cudaChannelFormatDesc channelDesc_A= cudaCreateChannelDesc<int>();
   const cudaExtent extent_A= make_cudaExtent(300, 300, 299);
   cudaMalloc3DArray(&cuArr_A, &channelDesc_A, extent_A);
   cudaMemcpy3DParms copyParams_A_to_device= {0};
   copyParams_A_to_device.srcPtr = make_cudaPitchedPtr((void*)A, extent_A.width*sizeof(int), extent_A.width, extent_A.height);
   copyParams_A_to_device.dstArray = cuArr_A;
   copyParams_A_to_device.extent = extent_A;
   copyParams_A_to_device.kind = cudaMemcpyHostToDevice;
   cudaMemcpy3D(&copyParams_A_to_device);
   cudaBindTextureToArray(texRef_A, cuArr_A, channelDesc_A);
   
   cudaArray* cuArr_B;
   cudaChannelFormatDesc channelDesc_B= cudaCreateChannelDesc<int>();
   const cudaExtent extent_B= make_cudaExtent(300, 300, 300);
   cudaMalloc3DArray(&cuArr_B, &channelDesc_B, extent_B, cudaArraySurfaceLoadStore);
   cudaMemcpy3DParms copyParams_B_to_device= {0};
   copyParams_B_to_device.srcPtr = make_cudaPitchedPtr((void*)B, extent_B.width*sizeof(int), extent_B.width, extent_B.height);
   copyParams_B_to_device.dstArray = cuArr_B;
   copyParams_B_to_device.extent = extent_B;
   copyParams_B_to_device.kind = cudaMemcpyHostToDevice;
   cudaMemcpy3D(&copyParams_B_to_device);
   cudaBindSurfaceToArray(surfRef_B, cuArr_B, channelDesc_B);
   
   {
     dim3 k0_dimBlock(4, 4, 32);
     dim3 k0_dimGrid(10, 10);
     kernel0 <<<k0_dimGrid, k0_dimBlock>>> ();
     cudaCheckKernel();
   }
   
   cudaMemcpy3DParms copyParams_B_from_device= {0};
   copyParams_B_from_device.dstPtr = make_cudaPitchedPtr((void*)B, extent_B.width*sizeof(int), extent_B.width, extent_B.height);
   copyParams_B_from_device.srcArray = cuArr_B;
   copyParams_B_from_device.extent = extent_B;
   copyParams_B_from_device.kind = cudaMemcpyDeviceToHost;
   cudaMemcpy3D(&copyParams_B_from_device);
   
   
   cudaUnbindTexture(texRef_A);
   
   cudaFreeArray(cuArr_A);
   cudaFreeArray(cuArr_B);
 }
  return 0;
}

