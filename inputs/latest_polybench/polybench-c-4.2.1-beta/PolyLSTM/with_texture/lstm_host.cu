#include <assert.h>
#include <stdio.h>
#define HOSTCODE true 
#include "lstm_kernel.hu"
/**
 * This version is stamped on Apr. 14, 2015
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 *
 *
 *
 *  FIXME : Update reference here.
 *
 *
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is N=1024. */
#include "lstm.h"

/* Array initialization. */
	static
void init_array(int nt, int np, int ns, int nq,
		DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
		DATA_TYPE POLYBENCH_2D(c_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(U_i,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_f,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_o,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_g,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(W_i,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_f,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_o,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_g,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_1D(i,NS,ns),
		DATA_TYPE POLYBENCH_1D(f,NS,ns),
		DATA_TYPE POLYBENCH_1D(o,NS,ns),
		DATA_TYPE POLYBENCH_1D(g,NS,ns),
		DATA_TYPE POLYBENCH_2D(del_S,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(del_C,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(del_Ui,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Uf,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Uo,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Ug,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Wi,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(del_Wf,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(del_Wo,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(del_Wg,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_1D(del_i,NS,ns),
		DATA_TYPE POLYBENCH_1D(del_f,NS,ns),
		DATA_TYPE POLYBENCH_1D(del_o,NS,ns),
		DATA_TYPE POLYBENCH_1D(del_g,NS,ns))
{
	int a, b;

	for (a = 0; a < nt; a++)
		for (b = 0; b < ns; b++) 
		{
			s_F[a][b] = (DATA_TYPE) ((a*b) % nt);
			del_S[a][b] = (DATA_TYPE) ((a*(b+2)) % ns);
			del_C[a][b] = (DATA_TYPE) (((a+2)*(b+5)) % nt);
			c_F[a][b] = (DATA_TYPE) ((a*b) % ns);
		}	

	for (a = 0; a < ns; a++)
		for (b = 0; b < np; b++) 
		{
			U_i[a][b] = (DATA_TYPE) ((a*b) % np);
			del_Ui[a][b] = (DATA_TYPE) ((a*(b+1)) % np);
			del_Uf[a][b] = (DATA_TYPE) ((a*(b+3)) % np);
			del_Uo[a][b] = (DATA_TYPE) ((a*(b+1)) % ns);
			del_Ug[a][b] = (DATA_TYPE) ((a*(b+3)) % ns);
			U_f[a][b] = (DATA_TYPE) ((a*(b+1)) % ns);
			U_o[a][b] = (DATA_TYPE) (((a+1)*(b+1)) % ns);
			U_g[a][b] = (DATA_TYPE) ((a*b) % np);
		}	

	for (a = 0; a < ns; a++)
		for (b = 0; b < ns; b++) 
		{
			W_i[a][b] = (DATA_TYPE) ((a*b) % ns);
			W_f[a][b] = (DATA_TYPE) ((a*(b+1)) % np);
			W_o[a][b] = (DATA_TYPE) (((a+1)*(b)) % nq);
			W_g[a][b] = (DATA_TYPE) (((a+1)*(b+1)) % nt);
			del_Wi[a][b] = (DATA_TYPE) ((a*b % ns));
			del_Wf[a][b] = (DATA_TYPE) (((a)*(b+1)) % np);
			del_Wo[a][b] = (DATA_TYPE) (((a+1)*b) % nq);
			del_Wg[a][b] = (DATA_TYPE) (((a+1)*(b+1)) % nt);
		}

	for (b = 0; b < ns; b++) 
	{
		i[b] = (DATA_TYPE) ((DATA_TYPE) b+ 1) / nt;
		f[b] = (DATA_TYPE) ((DATA_TYPE) b+ 7) / ns;
		o[b] = (DATA_TYPE) ((DATA_TYPE) b+ 3) / np;
		g[b] = (DATA_TYPE) ((DATA_TYPE) b+ 5) / nq;
	}

	for (a = 0; a < nt; a++)
		for (b = 0; b < np; b++) 
			inp_F[a][b] = (DATA_TYPE) ((a*b) % np);

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
	static
void print_array(int ns, DATA_TYPE POLYBENCH_1D(o,NS,ns))
{
	int a;

	for (a = 0; a < ns; a++)
	{
		fprintf (stderr, DATA_PRINTF_MODIFIER, o[a]);
		if ((a) % 20 == 0) fprintf (stderr, "\n");
	}
	fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
	static
void lstm_forward(int nt, int np, int ns, int nq,            
		DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
		DATA_TYPE POLYBENCH_2D(c_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(U_i,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_f,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_o,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_g,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(W_i,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_f,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_o,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_g,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_1D(i,NS,ns),
		DATA_TYPE POLYBENCH_1D(f,NS,ns),
		DATA_TYPE POLYBENCH_1D(o,NS,ns),
		DATA_TYPE POLYBENCH_1D(g,NS,ns))      
{
	int t, p, q, s1, s2, b;
	#define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
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

	  float (*dev_c_F)[2850];
	  float *dev_f;
	  float *dev_g;
	  float *dev_i;
	  float *dev_o;
	  float (*dev_s_F)[2850];
	  
	  cudaCheckReturn(cudaMalloc((void **) &dev_c_F, (400) * (2850) * sizeof(float)));
	  #define ppcg_max(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x > _y ? _x : _y; })
	  cudaCheckReturn(cudaMalloc((void **) &dev_f, (ppcg_max(2850, ns)) * sizeof(float)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_g, (ppcg_max(2850, ns)) * sizeof(float)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_i, (ppcg_max(2850, ns)) * sizeof(float)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_o, (ppcg_max(2850, ns)) * sizeof(float)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_s_F, (ns <= 0 ? 399 : 400) * (2850) * sizeof(float)));
	  
	  
	  cudaArray* cuArr_U_f;
	  cudaChannelFormatDesc channelDesc_U_f= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_U_f, &channelDesc_U_f, 3000, 2850);
	  cudaMemcpyToArray(cuArr_U_f, 0, 0, U_f, (2850) * (3000) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_U_f, cuArr_U_f, channelDesc_U_f);
	  
	  cudaArray* cuArr_U_g;
	  cudaChannelFormatDesc channelDesc_U_g= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_U_g, &channelDesc_U_g, 3000, 2850);
	  cudaMemcpyToArray(cuArr_U_g, 0, 0, U_g, (2850) * (3000) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_U_g, cuArr_U_g, channelDesc_U_g);
	  
	  cudaArray* cuArr_U_i;
	  cudaChannelFormatDesc channelDesc_U_i= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_U_i, &channelDesc_U_i, 3000, 2850);
	  cudaMemcpyToArray(cuArr_U_i, 0, 0, U_i, (2850) * (3000) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_U_i, cuArr_U_i, channelDesc_U_i);
	  
	  cudaArray* cuArr_U_o;
	  cudaChannelFormatDesc channelDesc_U_o= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_U_o, &channelDesc_U_o, 3000, 2850);
	  cudaMemcpyToArray(cuArr_U_o, 0, 0, U_o, (2850) * (3000) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_U_o, cuArr_U_o, channelDesc_U_o);
	  
	  cudaArray* cuArr_W_f;
	  cudaChannelFormatDesc channelDesc_W_f= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_W_f, &channelDesc_W_f, 2850, 2850);
	  cudaMemcpyToArray(cuArr_W_f, 0, 0, W_f, (2850) * (2850) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_W_f, cuArr_W_f, channelDesc_W_f);
	  
	  cudaArray* cuArr_W_g;
	  cudaChannelFormatDesc channelDesc_W_g= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_W_g, &channelDesc_W_g, 2850, 2850);
	  cudaMemcpyToArray(cuArr_W_g, 0, 0, W_g, (2850) * (2850) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_W_g, cuArr_W_g, channelDesc_W_g);
	  
	  cudaArray* cuArr_W_i;
	  cudaChannelFormatDesc channelDesc_W_i= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_W_i, &channelDesc_W_i, 2850, 2850);
	  cudaMemcpyToArray(cuArr_W_i, 0, 0, W_i, (2850) * (2850) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_W_i, cuArr_W_i, channelDesc_W_i);
	  
	  cudaArray* cuArr_W_o;
	  cudaChannelFormatDesc channelDesc_W_o= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_W_o, &channelDesc_W_o, 2850, 2850);
	  cudaMemcpyToArray(cuArr_W_o, 0, 0, W_o, (2850) * (2850) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_W_o, cuArr_W_o, channelDesc_W_o);
	  
	  cudaArray* cuArr_inp_F;
	  cudaChannelFormatDesc channelDesc_inp_F= cudaCreateChannelDesc<float>();
	  cudaMallocArray(&cuArr_inp_F, &channelDesc_inp_F, 3000, 400);
	  cudaMemcpyToArray(cuArr_inp_F, 0, 0, inp_F, (400) * (3000) * sizeof(float), cudaMemcpyHostToDevice);
	  cudaBindTextureToArray(texRef_inp_F, cuArr_inp_F, channelDesc_inp_F);
	  
	  if (ns >= 1)
	    cudaCheckReturn(cudaMemcpy(dev_c_F, c_F, (400) * (2850) * sizeof(float), cudaMemcpyHostToDevice));
	  cudaCheckReturn(cudaMemcpy(dev_f, f, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyHostToDevice));
	  cudaCheckReturn(cudaMemcpy(dev_g, g, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyHostToDevice));
	  cudaCheckReturn(cudaMemcpy(dev_i, i, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyHostToDevice));
	  cudaCheckReturn(cudaMemcpy(dev_o, o, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyHostToDevice));
	  cudaCheckReturn(cudaMemcpy(dev_s_F, s_F, (ns <= 0 ? 399 : 400) * (2850) * sizeof(float), cudaMemcpyHostToDevice));
	  for (int c0 = 0; c0 <= 400; c0 += 1) {
	    if (ns >= 1 && c0 >= 1)
	      {
	        dim3 k0_dimBlock(32);
	        dim3 k0_dimGrid(ppcg_min(32768, (ns + 31) / 32));
	        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_c_F, dev_o, dev_s_F, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	    if (c0 <= 399) {
	      {
	        dim3 k1_dimBlock(32);
	        dim3 k1_dimGrid(90);
	        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_i, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      {
	        dim3 k2_dimBlock(32);
	        dim3 k2_dimGrid(90);
	        kernel2 <<<k2_dimGrid, k2_dimBlock>>> (dev_f, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      {
	        dim3 k3_dimBlock(32);
	        dim3 k3_dimGrid(90);
	        kernel3 <<<k3_dimGrid, k3_dimBlock>>> (dev_g, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      {
	        dim3 k4_dimBlock(32);
	        dim3 k4_dimGrid(90);
	        kernel4 <<<k4_dimGrid, k4_dimBlock>>> (dev_o, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      {
	        dim3 k5_dimBlock(32);
	        dim3 k5_dimGrid(90);
	        kernel5 <<<k5_dimGrid, k5_dimBlock>>> (dev_f, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      {
	        dim3 k6_dimBlock(32);
	        dim3 k6_dimGrid(90);
	        kernel6 <<<k6_dimGrid, k6_dimBlock>>> (dev_g, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      {
	        dim3 k7_dimBlock(32);
	        dim3 k7_dimGrid(90);
	        kernel7 <<<k7_dimGrid, k7_dimBlock>>> (dev_i, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      if (c0 >= 1)
	        {
	          dim3 k8_dimBlock(32);
	          dim3 k8_dimGrid(90);
	          kernel8 <<<k8_dimGrid, k8_dimBlock>>> (dev_f, dev_s_F, ns, c0);
	          cudaCheckKernel();
	        }
	        
	        
	      {
	        dim3 k9_dimBlock(32);
	        dim3 k9_dimGrid(90);
	        kernel9 <<<k9_dimGrid, k9_dimBlock>>> (dev_o, ns, c0);
	        cudaCheckKernel();
	      }
	      
	      
	      if (c0 >= 1) {
	        {
	          dim3 k10_dimBlock(32);
	          dim3 k10_dimGrid(90);
	          kernel10 <<<k10_dimGrid, k10_dimBlock>>> (dev_i, dev_s_F, ns, c0);
	          cudaCheckKernel();
	        }
	        
	        
	        {
	          dim3 k11_dimBlock(32);
	          dim3 k11_dimGrid(90);
	          kernel11 <<<k11_dimGrid, k11_dimBlock>>> (dev_g, dev_s_F, ns, c0);
	          cudaCheckKernel();
	        }
	        
	        
	        {
	          dim3 k12_dimBlock(32);
	          dim3 k12_dimGrid(90);
	          kernel12 <<<k12_dimGrid, k12_dimBlock>>> (dev_o, dev_s_F, ns, c0);
	          cudaCheckKernel();
	        }
	        
	        
	        if (ns >= 1)
	          {
	            dim3 k13_dimBlock(32);
	            dim3 k13_dimGrid(ppcg_min(32768, (ns + 31) / 32));
	            kernel13 <<<k13_dimGrid, k13_dimBlock>>> (dev_c_F, dev_f, dev_g, dev_i, ns, c0);
	            cudaCheckKernel();
	          }
	          
	          
	      }
	    }
	  }
	  if (ns >= 1)
	    cudaCheckReturn(cudaMemcpy(c_F, dev_c_F, (400) * (2850) * sizeof(float), cudaMemcpyDeviceToHost));
	  cudaCheckReturn(cudaMemcpy(f, dev_f, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyDeviceToHost));
	  cudaCheckReturn(cudaMemcpy(g, dev_g, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyDeviceToHost));
	  cudaCheckReturn(cudaMemcpy(i, dev_i, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyDeviceToHost));
	  cudaCheckReturn(cudaMemcpy(o, dev_o, (ppcg_max(2850, ns)) * sizeof(float), cudaMemcpyDeviceToHost));
	  cudaCheckReturn(cudaMemcpy(s_F, dev_s_F, (ns <= 0 ? 399 : 400) * (2850) * sizeof(float), cudaMemcpyDeviceToHost));
	  
	  cudaUnbindTexture(texRef_U_f);
	  cudaUnbindTexture(texRef_U_g);
	  cudaUnbindTexture(texRef_U_i);
	  cudaUnbindTexture(texRef_U_o);
	  cudaUnbindTexture(texRef_W_f);
	  cudaUnbindTexture(texRef_W_g);
	  cudaUnbindTexture(texRef_W_i);
	  cudaUnbindTexture(texRef_W_o);
	  cudaUnbindTexture(texRef_inp_F);
	  
	  cudaFreeArray(cuArr_U_f);
	  cudaFreeArray(cuArr_U_g);
	  cudaFreeArray(cuArr_U_i);
	  cudaFreeArray(cuArr_U_o);
	  cudaFreeArray(cuArr_W_f);
	  cudaFreeArray(cuArr_W_g);
	  cudaFreeArray(cuArr_W_i);
	  cudaFreeArray(cuArr_W_o);
	  cudaFreeArray(cuArr_inp_F);
	  cudaCheckReturn(cudaFree(dev_c_F));
	  cudaCheckReturn(cudaFree(dev_f));
	  cudaCheckReturn(cudaFree(dev_g));
	  cudaCheckReturn(cudaFree(dev_i));
	  cudaCheckReturn(cudaFree(dev_o));
	  cudaCheckReturn(cudaFree(dev_s_F));
	}
}

/*	static
void lstm_backward(int nt, int np, int ns, int nq, 
		DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
		DATA_TYPE POLYBENCH_2D(c_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(U_i,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_f,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_o,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(U_g,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(W_i,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_f,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_o,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(W_g,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_1D(i,NS,ns),
		DATA_TYPE POLYBENCH_1D(f,NS,ns),
		DATA_TYPE POLYBENCH_1D(o,NS,ns),
		DATA_TYPE POLYBENCH_1D(g,NS,ns),
		DATA_TYPE POLYBENCH_2D(del_S,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(del_C,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(del_Ui,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Uf,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Uo,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Ug,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(del_Wi,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(del_Wf,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(del_Wo,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(del_Wg,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_1D(del_i,NS,ns),
		DATA_TYPE POLYBENCH_1D(del_f,NS,ns),
		DATA_TYPE POLYBENCH_1D(del_o,NS,ns),
		DATA_TYPE POLYBENCH_1D(del_g,NS,ns))
{
	int t, p, q, s, r;
#pragma scop


	for (t = _PB_T - 1; t > 0; t--)
	{  
		for(s = 0; s < _PB_S; s++)
			del_o[s] = c_F[t][s] * del_S[t][s];

		for(s = 0; s < _PB_S; s++)
			del_C[t][s] += o[s] * del_S[t][s];

		if(t > 0)
			for(s = 0; s < _PB_S; s++)
				del_f[s] = c_F[t-1][s] * del_C[t][s];

		if(t > 0)
			for(s = 0; s < _PB_S; s++)
				del_C[t-1][s] = f[s] * del_C[t][s];

		for(s = 0; s < _PB_S; s++)
			del_i[s] = g[s] * del_C[t][s];

		for(s = 0; s < _PB_S; s++)
			del_g[s] = i[s] * del_C[t][s];

		for(s = 0; s < _PB_S; s++){
			for(p = 0; p < _PB_P; p++){
				del_Ui[s][p] = inp_F[t][p] * i[s];
			}
		}

		for(s = 0; s < _PB_S; s++){
			for(p = 0; p < _PB_P; p++){
				del_Uf[s][p] = inp_F[t][p] * f[s];
			}
		}

		for(s = 0; s < _PB_S; s++){
			for(p = 0; p < _PB_P; p++){
				del_Uo[s][p] = inp_F[t][p] * o[s];
			}
		}

		for(s = 0; s < _PB_S; s++){
			for(p = 0; p < _PB_P; p++){
				del_Ug[s][p] = inp_F[t][p] * g[s];
			}
		}

		if(t > 0){
			for(s = 0; s < _PB_S; s++){
				for(r = 0; r < _PB_S; r++){
					del_Wi[s][r] = s_F[t-1][r] * i[s];
				}
			}
		}

		if(t > 0){
			for(s = 0; s < _PB_S; s++){
				for(r = 0; r < _PB_S; r++){
					del_Wf[s][r] = s_F[t-1][r] * f[s];
				}
			}
		}

		if(t > 0){
			for(s = 0; s < _PB_S; s++){
				for(r = 0; r < _PB_S; r++){
					del_Wo[s][r] = s_F[t-1][r] * o[s];
				}
			}
		}

		if(t > 0){
			for(s = 0; s < _PB_S; s++){
				for(r = 0; r < _PB_S; r++){
					del_Wg[s][r] = s_F[t-1][r] * g[s];
				}
			}
		}

		if(t > 0){
			for(s = 0; s < _PB_S; s++){
				for(r = 0; r < _PB_S; r++){
					del_S[t-1][r] = W_i[r][s] * del_i[s] + W_f[r][s] * del_f[s] + W_o[r][s] * del_o[s] + W_g[r][s] * del_g[s];
				}
			}
		}
	}
#pragma endscop
}
*/


int main(int argc, char** argv)
{
	/* Retrieve problem size. 
	   x - Input sequence         nt x np
	   s - State at each step     nt x ns
	   o - Output at each step    nt x nq
	   U - Matrix multiplied with x       ns x np 
	   W - Matrix multiplied with s(t-1)  ns x ns
	   V - Matrix multiplied with s(t)    ns x nq
	 */
	int nt = NT;
	int np = NP;
	int nq = NQ;
	int ns = NS;

	/* Variable declaration/allocation. */
	POLYBENCH_2D_ARRAY_DECL(s_F,DATA_TYPE,NT,NS,nt,ns);
	POLYBENCH_2D_ARRAY_DECL(inp_F,DATA_TYPE,NT,NP,nt,np);
	POLYBENCH_2D_ARRAY_DECL(c_F,DATA_TYPE,NT,NS,nt,ns);
	POLYBENCH_2D_ARRAY_DECL(U_i,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(U_f,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(U_o,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(U_g,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(W_i,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_2D_ARRAY_DECL(W_f,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_2D_ARRAY_DECL(W_o,DATA_TYPE,NS,NS,ns,ns); 
	POLYBENCH_2D_ARRAY_DECL(W_g,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_1D_ARRAY_DECL(i,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(f,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(o,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(g,DATA_TYPE,NS,ns);
	POLYBENCH_2D_ARRAY_DECL(del_S,DATA_TYPE,NT,NS,nt,ns);
	POLYBENCH_2D_ARRAY_DECL(del_C,DATA_TYPE,NT,NS,nt,ns);
	POLYBENCH_2D_ARRAY_DECL(del_Ui,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(del_Uf,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(del_Uo,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(del_Ug,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(del_Wi,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_2D_ARRAY_DECL(del_Wf,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_2D_ARRAY_DECL(del_Wo,DATA_TYPE,NS,NS,ns,ns); 
	POLYBENCH_2D_ARRAY_DECL(del_Wg,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_1D_ARRAY_DECL(del_i,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(del_f,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(del_o,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(del_g,DATA_TYPE,NS,ns);

	/* Initialize array(s). */
	init_array(nt,np,ns,nq,
			POLYBENCH_ARRAY(s_F),
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(c_F),
			POLYBENCH_ARRAY(U_i),
			POLYBENCH_ARRAY(U_f),
			POLYBENCH_ARRAY(U_o),
			POLYBENCH_ARRAY(U_g),
			POLYBENCH_ARRAY(W_i),
			POLYBENCH_ARRAY(W_f),
			POLYBENCH_ARRAY(W_o),
			POLYBENCH_ARRAY(W_g),
			POLYBENCH_ARRAY(i),
			POLYBENCH_ARRAY(f),
			POLYBENCH_ARRAY(o),
			POLYBENCH_ARRAY(g),
			POLYBENCH_ARRAY(del_S),
			POLYBENCH_ARRAY(del_C),
			POLYBENCH_ARRAY(del_Ui),
			POLYBENCH_ARRAY(del_Uf),
			POLYBENCH_ARRAY(del_Uo),
			POLYBENCH_ARRAY(del_Ug),
			POLYBENCH_ARRAY(del_Wi),
			POLYBENCH_ARRAY(del_Wf),
			POLYBENCH_ARRAY(del_Wo),
			POLYBENCH_ARRAY(del_Wg),
			POLYBENCH_ARRAY(del_i),
			POLYBENCH_ARRAY(del_f),
			POLYBENCH_ARRAY(del_o),
			POLYBENCH_ARRAY(del_g));

	/* Start timer. */
	polybench_start_instruments;

	/* Run kernel. */
	lstm_forward(nt, np, ns, nq,
			POLYBENCH_ARRAY(s_F),
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(c_F),
			POLYBENCH_ARRAY(U_i),
			POLYBENCH_ARRAY(U_f),
			POLYBENCH_ARRAY(U_o),
			POLYBENCH_ARRAY(U_g),
			POLYBENCH_ARRAY(W_i),
			POLYBENCH_ARRAY(W_f),
			POLYBENCH_ARRAY(W_o),
			POLYBENCH_ARRAY(W_g),
			POLYBENCH_ARRAY(i),
			POLYBENCH_ARRAY(f),
			POLYBENCH_ARRAY(o),
			POLYBENCH_ARRAY(g));

	/*lstm_backward(nt, np, ns, nq,
	   POLYBENCH_ARRAY(s_F),
	   POLYBENCH_ARRAY(inp_F),
	   POLYBENCH_ARRAY(c_F),
	   POLYBENCH_ARRAY(U_i),
	   POLYBENCH_ARRAY(U_f),
	   POLYBENCH_ARRAY(U_o),
	   POLYBENCH_ARRAY(U_g),
	   POLYBENCH_ARRAY(W_i),
	   POLYBENCH_ARRAY(W_f),
	   POLYBENCH_ARRAY(W_o),
	   POLYBENCH_ARRAY(W_g),
	   POLYBENCH_ARRAY(i),
	   POLYBENCH_ARRAY(f),
	   POLYBENCH_ARRAY(o),
	   POLYBENCH_ARRAY(g) ,
	   POLYBENCH_ARRAY(del_S),
	   POLYBENCH_ARRAY(del_C),
	   POLYBENCH_ARRAY(del_Ui),
	   POLYBENCH_ARRAY(del_Uf),
	   POLYBENCH_ARRAY(del_Uo),
	   POLYBENCH_ARRAY(del_Ug),
	   POLYBENCH_ARRAY(del_Wi),
	   POLYBENCH_ARRAY(del_Wf),
	   POLYBENCH_ARRAY(del_Wo),
	   POLYBENCH_ARRAY(del_Wg),
	   POLYBENCH_ARRAY(del_i),
	   POLYBENCH_ARRAY(del_f),
	   POLYBENCH_ARRAY(del_o),
	   POLYBENCH_ARRAY(del_g));*/

	/* Stop and print timer. */
	polybench_stop_instruments;
	polybench_print_instruments;

	/* Prevent dead-code elimination. All live-out data must be printed
	   by the function call in argument. */
	polybench_prevent_dce(print_array(ns,  POLYBENCH_ARRAY(o)));

	/* Be clean. */
	POLYBENCH_FREE_ARRAY(s_F);
	POLYBENCH_FREE_ARRAY(inp_F);
	POLYBENCH_FREE_ARRAY(c_F);
	POLYBENCH_FREE_ARRAY(U_i);
	POLYBENCH_FREE_ARRAY(U_f);
	POLYBENCH_FREE_ARRAY(U_o);
	POLYBENCH_FREE_ARRAY(U_g);
	POLYBENCH_FREE_ARRAY(W_i);
	POLYBENCH_FREE_ARRAY(W_f);
	POLYBENCH_FREE_ARRAY(W_o);
	POLYBENCH_FREE_ARRAY(W_g);
	POLYBENCH_FREE_ARRAY(i);
	POLYBENCH_FREE_ARRAY(f);
	POLYBENCH_FREE_ARRAY(o);
	POLYBENCH_FREE_ARRAY(g);
	POLYBENCH_FREE_ARRAY(del_S);
	POLYBENCH_FREE_ARRAY(del_C);
	POLYBENCH_FREE_ARRAY(del_Ui);
	POLYBENCH_FREE_ARRAY(del_Uf);
	POLYBENCH_FREE_ARRAY(del_Uo);
	POLYBENCH_FREE_ARRAY(del_Ug);
	POLYBENCH_FREE_ARRAY(del_Wi);
	POLYBENCH_FREE_ARRAY(del_Wf);
	POLYBENCH_FREE_ARRAY(del_Wo);
	POLYBENCH_FREE_ARRAY(del_Wg);
	POLYBENCH_FREE_ARRAY(del_i);
	POLYBENCH_FREE_ARRAY(del_f);
	POLYBENCH_FREE_ARRAY(del_o);
	POLYBENCH_FREE_ARRAY(del_g);

	return 0;
}
