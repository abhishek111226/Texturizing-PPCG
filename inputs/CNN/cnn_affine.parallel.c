#include <omp.h>
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
  int t1, t2, t3, t4, t5, t6, t7;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NC >= 1) && (_PB_NK >= 1) && (_PB_NN >= 1) && (_PB_NP >= 1) && (_PB_NQ >= 1) && (_PB_NR >= 1) && (_PB_NS >= 1)) {
  lbp=0;
  ubp=_PB_NN-1;
#pragma omp parallel for private(lbv,ubv,t2,t3,t4,t5,t6,t7)
  for (t1=lbp;t1<=ubp;t1++) {
    for (t2=0;t2<=_PB_NK-1;t2++) {
      for (t3=0;t3<=_PB_NP-1;t3++) {
        for (t4=0;t4<=_PB_NQ-1;t4++) {
          for (t5=0;t5<=_PB_NC-1;t5++) {
            for (t6=0;t6<=_PB_NR-1;t6++) {
              for (t7=0;t7<=_PB_NS-1;t7++) {
                out_F[t1][t2][t3][t4] += W[t2][t5][t6][t7] * inp_F[t1][t5][5*t3+NR-t6-1][5*t4+NS-t7-1];;
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
  int t1, t2, t3, t4, t5, t6, t7, t8, t9;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NC >= 1) && (_PB_NH >= 1) && (_PB_NH >= -_PB_NR+NR+1) && (_PB_NK >= 1) && (_PB_NN >= 1) && (_PB_NP >= 1) && (5*_PB_NP >= -NR+6) && (_PB_NQ >= 1) && (5*_PB_NQ >= -NS+6) && (_PB_NR >= 1) && (_PB_NS >= 1) && (_PB_NW >= 1) && (_PB_NW >= -_PB_NS+NS+1)) {
  lbp=0;
  ubp=_PB_NN-1;
#pragma omp parallel for private(lbv,ubv,t2,t3,t4,t5,t6,t7,t8,t9)
  for (t1=lbp;t1<=ubp;t1++) {
    for (t2=0;t2<=_PB_NC-1;t2++) {
      for (t3=max(0,-_PB_NR+NR);t3<=min(_PB_NH-1,5*_PB_NP+NR-6);t3++) {
        for (t4=max(0,-_PB_NS+NS);t4<=min(_PB_NW-1,5*_PB_NQ+NS-6);t4++) {
          for (t5=0;t5<=_PB_NK-1;t5++) {
            for (t6=max(0,ceild(t3-NR+1,5));t6<=min(floord(t3+_PB_NR-NR,5),_PB_NP-1);t6++) {
              for (t8=max(0,ceild(t4-NS+1,5));t8<=min(floord(t4+_PB_NS-NS,5),_PB_NQ-1);t8++) {
                err_in[t1][t2][t3][t4] += W[t5][t2][(-t3+5*t6+NR-1)][(-t4+5*t8+NS-1)] * err_out[t1][t5][t6][t8];;
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
