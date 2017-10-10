#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

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
 *		
 *
 *	Reference : cuDNN paper.
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
#include "cnn.h"


/* Array initialization. */
	static
void init_array(int nn, int nk,int np,int nq,int nc,int nr,int ns,int nw,int nh, 
		DATA_TYPE POLYBENCH_4D(out_F,NN,NK,NP,NQ,nn,nk,np,nq),
		DATA_TYPE POLYBENCH_4D(W,NK,NC,NR,NS,nk,nc,nr,ns),
		DATA_TYPE POLYBENCH_4D(inp_F,NN,NC,NH,NW,nn,nc,nh,nw),
		DATA_TYPE POLYBENCH_4D(err_in,NN,NC,NH,NW,nn,nc,nh,nw),
		DATA_TYPE POLYBENCH_4D(err_out,NN,NK,NP,NQ,nn,nk,np,nq))
{
	int a, b, e, d;

	for (a = 0; a < nn; a++)
		for (b = 0; b < nk; b++) 
			for (e = 0; e < np; e++) 
				for (d = 0; d < nq; d++){
					out_F[a][b][e][d] = (DATA_TYPE) ((a*b) % nn);
					err_out[a][b][e][d] = (DATA_TYPE) ((e*d) % nk);
				}

	for (a = 0; a < nk; a++)
		for (b = 0; b < nc; b++) 
			for (e = 0; e < nr; e++) 
				for (d = 0; d < ns; d++)
					W[a][b][e][d] = (DATA_TYPE) ((a*b) % nc) / (10 * nc);

	for (a = 0; a < nn; a++)
		for (b = 0; b < nc; b++) 
			for (e = 0; e < nh; e++) 
				for (d = 0; d < nw; d++){
					inp_F[a][b][e][d] = (DATA_TYPE) ((a*b) % nc);	
					err_in[a][b][e][d]= (DATA_TYPE) ((a*b) % nc);
				}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
	static
void print_array(int nn, int nk, int np, int nq, DATA_TYPE POLYBENCH_4D(out_F,NN,NK,NP,NQ,nn,nk,np,nq))
{
	int a, b, e, d;
	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("out_F");
	for (a = 0; a < nn; a++)
		for (b = 0; b < nk; b++) 
			for (e = 0; e < np; e++) 
				for (d = 0; d < nq; d++) 
				{
					fprintf (stderr, DATA_PRINTF_MODIFIER, out_F[a][b][e][d]);
					if ((a*nk*np*nq + b * np * nq + e *nq + d) % 20 == 0) fprintf (stderr, "\n");
				}
	POLYBENCH_DUMP_END("out_F");
	POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
	static
void cnn_forward(int nn, int nk,int np,int nq,int nc,int nr,int ns,int nw,int nh,int u,int v,  
		DATA_TYPE POLYBENCH_4D(out_F,NN,NK,NP,NQ,nn,nk,np,nq),
		DATA_TYPE POLYBENCH_4D(W,NK,NC,NR,NS,nk,nc,nr,ns),
		DATA_TYPE POLYBENCH_4D(inp_F,NN,NC,NH,NW,nn,nc,nh,nw))
{
	int n, k, p, q, c, r, s;
/* Copyright (C) 1991-2015 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses ISO/IEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0.  */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NC >= 1) && (_PB_NK >= 1) && (_PB_NN >= 1) && (_PB_NP >= 1) && (_PB_NQ >= 1) && (_PB_NR >= 1) && (_PB_NS >= 1)) {
  for (t1=0;t1<=floord(_PB_NN-1,32);t1++) {
    for (t2=0;t2<=floord(_PB_NK-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NP-1,32);t3++) {
        for (t4=0;t4<=floord(_PB_NQ-1,32);t4++) {
          for (t5=0;t5<=floord(_PB_NC-1,32);t5++) {
            for (t6=32*t1;t6<=min(_PB_NN-1,32*t1+31);t6++) {
              for (t7=32*t2;t7<=min(_PB_NK-1,32*t2+31);t7++) {
                for (t8=32*t3;t8<=min(_PB_NP-1,32*t3+31);t8++) {
                  for (t9=32*t4;t9<=min(_PB_NQ-1,32*t4+31);t9++) {
                    for (t10=32*t5;t10<=min(_PB_NC-1,32*t5+31);t10++) {
                      for (t11=0;t11<=_PB_NR-1;t11++) {
                        for (t12=0;t12<=_PB_NS-1;t12++) {
                          out_F[t6][t7][t8][t9] += W[t7][t10][t11][t12] * inp_F[t6][t10][5*t8+NR-t11-1][5*t9+NS-t12-1];;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/* End of CLooG code */
}

void cnn_backward(int nn, int nk,int np,int nq,int nc,int nr,int ns,int nw,int nh,int u,int v,  
		DATA_TYPE POLYBENCH_4D(err_out,NN,NK,NP,NQ,nn,nk,np,nq),
		DATA_TYPE POLYBENCH_4D(W,NK,NC,NR,NS,nk,nc,nr,ns),
		DATA_TYPE POLYBENCH_4D(err_in,NN,NC,NH,NW,nn,nc,nh,nw))
{

	int n, k, h, w, p, q, c, r, s;
/* Copyright (C) 1991-2015 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses ISO/IEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0.  */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NC >= 1) && (_PB_NH >= 1) && (_PB_NH >= -_PB_NR+NR+1) && (_PB_NK >= 1) && (_PB_NN >= 1) && (_PB_NP >= 1) && (5*_PB_NP >= -NR+6) && (_PB_NQ >= 1) && (5*_PB_NQ >= -NS+6) && (_PB_NR >= 1) && (_PB_NS >= 1) && (_PB_NW >= 1) && (_PB_NW >= -_PB_NS+NS+1)) {
  for (t1=0;t1<=floord(_PB_NN-1,32);t1++) {
    for (t2=0;t2<=floord(_PB_NC-1,32);t2++) {
      for (t3=max(0,ceild(-_PB_NR+NR-31,32));t3<=min(floord(_PB_NH-1,32),floord(5*_PB_NP+NR-6,32));t3++) {
        for (t4=max(0,ceild(-_PB_NS+NS-31,32));t4<=min(floord(_PB_NW-1,32),floord(5*_PB_NQ+NS-6,32));t4++) {
          for (t5=0;t5<=floord(_PB_NK-1,32);t5++) {
            for (t6=32*t1;t6<=min(_PB_NN-1,32*t1+31);t6++) {
              for (t7=32*t2;t7<=min(_PB_NC-1,32*t2+31);t7++) {
                for (t8=max(32*t3,-_PB_NR+NR);t8<=min(min(_PB_NH-1,32*t3+31),5*_PB_NP+NR-6);t8++) {
                  for (t9=max(32*t4,-_PB_NS+NS);t9<=min(min(_PB_NW-1,32*t4+31),5*_PB_NQ+NS-6);t9++) {
                    for (t10=32*t5;t10<=min(_PB_NK-1,32*t5+31);t10++) {
                      for (t11=max(0,ceild(t8-NR+1,5));t11<=min(floord(t8+_PB_NR-NR,5),_PB_NP-1);t11++) {
                        for (t13=max(0,ceild(t9-NS+1,5));t13<=min(floord(t9+_PB_NS-NS,5),_PB_NQ-1);t13++) {
                          err_in[t6][t7][t8][t9] += W[t10][t7][(-t8+5*t11+NR-1)][(-t9+5*t13+NS-1)] * err_out[t6][t10][t11][t13];;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/* End of CLooG code */
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. 
	   nn -> Batch size
	   nk -> Number of output feature maps
	   np -> Output matrix height
	   nq -> Output matrix width
	   nc -> Number of input feature maps
	   nr -> Filter height
	   ns -> Filter width
	   nh -> Input matrix height
	   nw -> Input matrix width
	 */
	int nn = NN;	
	int nk = NK;
	int np = NP;
	int nq = NQ;
	int nc = NC;
	int nr = NR;
	int ns = NS;
	int nw = NW;
	int nh = NH;
	int nu = NU;
	int nv = NV;	

	/* Variable declaration/allocation. */
	POLYBENCH_4D_ARRAY_DECL(out_F,DATA_TYPE,NN,NK,NP,NQ,nn,nk,np,nq);
	POLYBENCH_4D_ARRAY_DECL(W,DATA_TYPE,NK,NC,NR,NS,nk,nc,nr,ns);
	POLYBENCH_4D_ARRAY_DECL(inp_F,DATA_TYPE,NN,NC,NH,NW,nn,nc,nh,nw);
	POLYBENCH_4D_ARRAY_DECL(err_in,DATA_TYPE,NN,NC,NH,NW,nn,nc,nh,nw);
	POLYBENCH_4D_ARRAY_DECL(err_out,DATA_TYPE,NN,NK,NP,NQ,nn,nk,np,nq);

	/* Initialize array(s). */
	init_array (nn,nk,np,nq,nc,nr,ns,nw,nh,
			POLYBENCH_ARRAY(out_F),
			POLYBENCH_ARRAY(W),
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(err_in),
			POLYBENCH_ARRAY(err_out));


	/* Start timer. */
	polybench_start_instruments;

	/* Run kernel. */
	cnn_forward(nn, nk, np, nq, nc, nr, ns, nw, nh, nu, nv,
			POLYBENCH_ARRAY(out_F),
			POLYBENCH_ARRAY(W),
			POLYBENCH_ARRAY(inp_F));

	cnn_backward(nn, nk, np, nq, nc, nr, ns, nw, nh, nu, nv,
			POLYBENCH_ARRAY(err_out),
			POLYBENCH_ARRAY(W),
			POLYBENCH_ARRAY(err_in));

	/* Stop and print timer. */
	polybench_stop_instruments;
	polybench_print_instruments;

	/* Prevent dead-code elimination. All live-out data must be printed
	   by the function call in argument. */
	polybench_prevent_dce(print_array(nn, nk, np, nq,  POLYBENCH_ARRAY(out_F)));

	/* Be clean. */
	POLYBENCH_FREE_ARRAY(out_F);
	POLYBENCH_FREE_ARRAY(W);
	POLYBENCH_FREE_ARRAY(inp_F);
	POLYBENCH_FREE_ARRAY(err_out);
	POLYBENCH_FREE_ARRAY(err_in);

	return 0;
}
