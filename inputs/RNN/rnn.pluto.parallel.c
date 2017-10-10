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
 *      Implemented by referring link.
 *       http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/09/rnn.jpg
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
#include "rnn.h"


/* Array initialization. */
	static
void init_array(int nt, int np, int ns, int nq,
		DATA_TYPE POLYBENCH_2D(out_F,NT,NQ,nt,nq),
		DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
		DATA_TYPE POLYBENCH_2D(U,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(W,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(V,NQ,NS,nq,ns),
		DATA_TYPE POLYBENCH_2D(err_out,NT,NQ,nt,nq),
		DATA_TYPE POLYBENCH_2D(delU,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(delW,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(delV,NQ,NS,nq,ns),
		DATA_TYPE POLYBENCH_ARRAY(delTA),
		DATA_TYPE POLYBENCH_ARRAY(delTB))
{
	int a, b;

	for (a = 0; a < nt; a++)
		for (b = 0; b < nq; b++) 
			out_F[a][b] = (DATA_TYPE) (a*b % nq) / nq;

	for (a = 0; a < nt; a++)
		for (b = 0; b < ns; b++) 
			s_F[a][b] = (DATA_TYPE) (a*b % nt) / nt;

	for (a = 0; a < nt; a++)
		for (b = 0; b < np; b++) 
			inp_F[a][b] = (DATA_TYPE) (a*b % np ) / np;

	for (a = 0; a < ns; a++)
		for (b = 0; b < np; b++) 
			U[a][b] = (DATA_TYPE) (a*b % nt) / nt;

	for (a = 0; a < ns; a++)
		for (b = 0; b < ns; b++) 
			W[a][b] = (DATA_TYPE) (a*b % ns) / ns;

	for (a = 0; a < nq; a++)
		for (b = 0; b < ns; b++) 
			V[a][b] = (DATA_TYPE) ((a+1)*b  % nq) / nq;

	for (a = 0; a < nt; a++)
		for (b = 0; b < nq; b++) 
			err_out[a][b] = (DATA_TYPE) (a*(b+1)  % nt) / nt;

	for (a = 0; a < ns; a++)
		for (b = 0; b < np; b++) 
			delU[a][b] = (DATA_TYPE) ((a+1)*(b+1) % nt) / nt;

	for (a = 0; a < ns; a++)
		for (b = 0; b < ns; b++) 
			delW[a][b] = (DATA_TYPE) ((a+1)*(b+1) % nt) / nt ;

	for (a = 0; a < nq; a++)
		for (b = 0; b < ns; b++) 
			delV[a][b] = (DATA_TYPE) ((a+1)*(b+1) % nt) / nt;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
	static
void print_array_fwd(int nt, int nq, DATA_TYPE POLYBENCH_2D(out_F,NT,NQ,nt,nq))
{
	int a, b, e, d;
	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("out_F");
	for (a = 0; a < nt; a++)
		for (b = 0; b < nq; b++) 
		{
			fprintf (stderr, DATA_PRINTF_MODIFIER, out_F[a][b]);
			if ((a*nq+ b) % 20 == 0) fprintf (stderr, "\n");
		}
	POLYBENCH_DUMP_END("out_F");
	POLYBENCH_DUMP_FINISH;
}


	static
void print_array_bwd(int np, int nq, int ns, DATA_TYPE  POLYBENCH_2D(delU,NS,NP,ns,np), 
		DATA_TYPE POLYBENCH_2D(delW, NS,NS,ns,ns), DATA_TYPE POLYBENCH_2D(delV, NQ,NS,nq,ns), 
		DATA_TYPE POLYBENCH_1D(delTA, NS,ns))
{
	int a, b, e, d;
	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("delU");
	for (a = 0; a < ns; a++)
		for (b = 0; b < np; b++) 
		{
			fprintf (stderr, DATA_PRINTF_MODIFIER, delU[a][b]);
			if ((a*np+ b) % 20 == 0) fprintf (stderr, "\n");
		}
	POLYBENCH_DUMP_END("delU");
	POLYBENCH_DUMP_FINISH;
	fprintf (stderr, "\n");

	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("delW");
	for (a = 0; a < ns; a++)
		for (b = 0; b < ns; b++) 
		{
			fprintf (stderr, DATA_PRINTF_MODIFIER, delW[a][b]);
			if ((a*ns+ b) % 20 == 0) fprintf (stderr, "\n");
		}
	POLYBENCH_DUMP_END("delW");
	POLYBENCH_DUMP_FINISH;
	fprintf (stderr, "\n");

	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("delV");
	for (a = 0; a < nq; a++)
		for (b = 0; b < ns; b++) 
		{
			fprintf (stderr, DATA_PRINTF_MODIFIER, delV[a][b]);
			if ((a*ns+ b) % 20 == 0) fprintf (stderr, "\n");
		}
	POLYBENCH_DUMP_END("delV");
	POLYBENCH_DUMP_FINISH;
	fprintf (stderr, "\n");

	POLYBENCH_DUMP_START;
	POLYBENCH_DUMP_BEGIN("delTA");
	for (a = 0; a < ns; a++)
	{
		fprintf (stderr, DATA_PRINTF_MODIFIER, delTA[a]);
		if (a % 20 == 0) fprintf (stderr, "\n");
	}
	POLYBENCH_DUMP_END("delTA");
	POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
	static
void rnn_forward(int nt, int np, int ns, int nq,            
		DATA_TYPE POLYBENCH_2D(out_F,NT,NQ,nt,nq),
		DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
		DATA_TYPE POLYBENCH_2D(U,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(W,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(V,NQ,NS,nq,ns))
{
	int t, p, q, s1, s2;
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
  int t1, t2, t3, t4;
 int lb, ub, lbp, ubp, lb2, ub2;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NS >= 1) && (_PB_NT >= 1)) {
  if (_PB_NP >= 1) {
    lbp=0;
    ubp=_PB_NT-1;
#pragma omp parallel for private(lbv,ubv,t3,t4)
    for (t2=lbp;t2<=ubp;t2++) {
      for (t3=0;t3<=_PB_NS-1;t3++) {
        for (t4=0;t4<=_PB_NP-1;t4++) {
          s_F[t2][t3] += U[t3][t4] * inp_F[t2][t4];;
        }
      }
    }
  }
  if (_PB_NT >= 2) {
    lbp=0;
    ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t3,t4)
    for (t2=lbp;t2<=ubp;t2++) {
      for (t3=1;t3<=_PB_NT-1;t3++) {
        s_F[t3][s1] += W[s1][t2] * s_F[t3-1][t2];;
      }
    }
  }
  if (_PB_NQ >= 1) {
    lbp=0;
    ubp=_PB_NT-1;
#pragma omp parallel for private(lbv,ubv,t3,t4)
    for (t2=lbp;t2<=ubp;t2++) {
      for (t3=0;t3<=_PB_NQ-1;t3++) {
        for (t4=0;t4<=_PB_NS-1;t4++) {
          out_F[t2][t3] += V[t3][t4] * s_F[t2][t4];;
        }
      }
    }
  }
}
/* End of CLooG code */
}


	static
void rnn_backward(int nt, int np, int ns, int nq, int bptt_trunc,  
		DATA_TYPE POLYBENCH_2D(inp_F,NT,NP,nt,np),
		DATA_TYPE POLYBENCH_2D(s_F,NT,NS,nt,ns),
		DATA_TYPE POLYBENCH_2D(W,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(V,NQ,NS,nq,ns),
		DATA_TYPE POLYBENCH_2D(err_out,NT,NQ,nt,nq),
		DATA_TYPE POLYBENCH_2D(delU,NS,NP,ns,np),
		DATA_TYPE POLYBENCH_2D(delW,NS,NS,ns,ns),
		DATA_TYPE POLYBENCH_2D(delV,NQ,NS,nq,ns),
		DATA_TYPE POLYBENCH_1D(delTA,NS,ns),
		DATA_TYPE POLYBENCH_1D(delTB,NS,ns))
{
	int t, p, q, r, s, step;
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
if ((_PB_NS >= 1) && (_PB_NT >= 2)) {
  if ((_PB_NP >= 1) && (_PB_NQ >= 1) && (bptt_trunc >= 0)) {
    for (t2=-_PB_NT+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_NQ-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          delV[t4][t6] = err_out[-t2][t4] * s_F[-t2][t6];;
        }
      }
      lbp=0;
      ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        delTA[t4] = 0;;
        for (t6=0;t6<=_PB_NQ-1;t6++) {
          delTA[t4] += V[t6][t4] * err_out[-t2][t6];;
        }
      }
      for (t4=t2-1;t4<=min(-1,t2+bptt_trunc-1);t4++) {
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          lbv=0;
          ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            delW[t6][t8] += delTA[t6] * s_F[-t4 - 1][t8];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          lbv=0;
          ubv=_PB_NP-1;
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            delU[t6][t8] += delTA[t6] * inp_F[-t4][t8];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTB[t6] = 0;;
          for (t8=0;t8<=_PB_NS-1;t8++) {
            delTB[t6] += delTA[t8] * W[t8][t6];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTA[t6] = delTB[t6];;
        }
      }
    }
  }
  if ((_PB_NP <= 0) && (_PB_NQ >= 1) && (bptt_trunc >= 0)) {
    for (t2=-_PB_NT+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_NQ-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          delV[t4][t6] = err_out[-t2][t4] * s_F[-t2][t6];;
        }
      }
      lbp=0;
      ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        delTA[t4] = 0;;
        for (t6=0;t6<=_PB_NQ-1;t6++) {
          delTA[t4] += V[t6][t4] * err_out[-t2][t6];;
        }
      }
      for (t4=t2-1;t4<=min(-1,t2+bptt_trunc-1);t4++) {
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          lbv=0;
          ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            delW[t6][t8] += delTA[t6] * s_F[-t4 - 1][t8];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTB[t6] = 0;;
          for (t8=0;t8<=_PB_NS-1;t8++) {
            delTB[t6] += delTA[t8] * W[t8][t6];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTA[t6] = delTB[t6];;
        }
      }
    }
  }
  if ((_PB_NQ >= 1) && (bptt_trunc <= -1)) {
    for (t2=-_PB_NT+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_NQ-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          delV[t4][t6] = err_out[-t2][t4] * s_F[-t2][t6];;
        }
      }
      lbp=0;
      ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        delTA[t4] = 0;;
        for (t6=0;t6<=_PB_NQ-1;t6++) {
          delTA[t4] += V[t6][t4] * err_out[-t2][t6];;
        }
      }
    }
  }
  if ((_PB_NP >= 1) && (_PB_NQ <= 0) && (bptt_trunc >= 0)) {
    for (t2=-_PB_NT+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        delTA[t4] = 0;;
      }
      for (t4=t2-1;t4<=min(-1,t2+bptt_trunc-1);t4++) {
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          lbv=0;
          ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            delW[t6][t8] += delTA[t6] * s_F[-t4 - 1][t8];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          lbv=0;
          ubv=_PB_NP-1;
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            delU[t6][t8] += delTA[t6] * inp_F[-t4][t8];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTB[t6] = 0;;
          for (t8=0;t8<=_PB_NS-1;t8++) {
            delTB[t6] += delTA[t8] * W[t8][t6];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTA[t6] = delTB[t6];;
        }
      }
    }
  }
  if ((_PB_NP <= 0) && (_PB_NQ <= 0) && (bptt_trunc >= 0)) {
    for (t2=-_PB_NT+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        delTA[t4] = 0;;
      }
      for (t4=t2-1;t4<=min(-1,t2+bptt_trunc-1);t4++) {
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          lbv=0;
          ubv=_PB_NS-1;
#pragma ivdep
#pragma vector always
          for (t8=lbv;t8<=ubv;t8++) {
            delW[t6][t8] += delTA[t6] * s_F[-t4 - 1][t8];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTB[t6] = 0;;
          for (t8=0;t8<=_PB_NS-1;t8++) {
            delTB[t6] += delTA[t8] * W[t8][t6];;
          }
        }
        lbp=0;
        ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t7,t8,t9)
        for (t6=lbp;t6<=ubp;t6++) {
          delTA[t6] = delTB[t6];;
        }
      }
    }
  }
  if ((_PB_NQ <= 0) && (bptt_trunc <= -1)) {
    for (t2=-_PB_NT+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_NS-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7,t8,t9)
      for (t4=lbp;t4<=ubp;t4++) {
        delTA[t4] = 0;;
      }
    }
  }
}
/* End of CLooG code */

}
int main(int argc, char** argv)
{
	/* Retrieve problem size. 
	   x - Input sequence         nt x np
	   s - State at each step     nt x ns
	   o - Output at each step    nt x nq
	   U - Matrix multiplied with x       ns x np 
	   W - Matrix multiplied with s(t-1)  ns x ns
	   V - Matrix multiplied with s(t)    nq x ns

	 */
	int nt = NT;
	int np = NP;
	int nq = NQ;
	int ns = NS;
	int bptt_trunc = BT;

	/* Variable declaration/allocation. */
	POLYBENCH_2D_ARRAY_DECL(out_F,DATA_TYPE,NT,NQ,nt,nq);
	POLYBENCH_2D_ARRAY_DECL(s_F,DATA_TYPE,NT,NS,nt,ns);
	POLYBENCH_2D_ARRAY_DECL(inp_F,DATA_TYPE,NT,NP,nt,np);
	POLYBENCH_2D_ARRAY_DECL(U,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(W,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_2D_ARRAY_DECL(V,DATA_TYPE,NQ,NS,nq,ns);
	POLYBENCH_2D_ARRAY_DECL(err_out,DATA_TYPE,NT,NQ,nt,nq);
	POLYBENCH_2D_ARRAY_DECL(delU,DATA_TYPE,NS,NP,ns,np);
	POLYBENCH_2D_ARRAY_DECL(delW,DATA_TYPE,NS,NS,ns,ns);
	POLYBENCH_2D_ARRAY_DECL(delV,DATA_TYPE,NQ,NS,nq,ns);
	POLYBENCH_1D_ARRAY_DECL(delTA,DATA_TYPE,NS,ns);
	POLYBENCH_1D_ARRAY_DECL(delTB,DATA_TYPE,NS,ns);


	/* Initialize array(s). */
	init_array(nt,np,ns,nq,
			POLYBENCH_ARRAY(out_F),
			POLYBENCH_ARRAY(s_F),
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(U),
			POLYBENCH_ARRAY(W),
			POLYBENCH_ARRAY(V),
			POLYBENCH_ARRAY(err_out),
			POLYBENCH_ARRAY(delU),
			POLYBENCH_ARRAY(delW),
			POLYBENCH_ARRAY(delV),
			POLYBENCH_ARRAY(delTA),
			POLYBENCH_ARRAY(delTB));

	/* Start timer. */
	polybench_start_instruments;

	/* Run kernel. */
	rnn_forward(nt, np, ns, nq,
			POLYBENCH_ARRAY(out_F),
			POLYBENCH_ARRAY(s_F),
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(U),
			POLYBENCH_ARRAY(W),
			POLYBENCH_ARRAY(V));

	/*rnn_backward(nt, np, ns, nq, bptt_trunc,  
			POLYBENCH_ARRAY(inp_F),
			POLYBENCH_ARRAY(s_F),
			POLYBENCH_ARRAY(W),
			POLYBENCH_ARRAY(V),
			POLYBENCH_ARRAY(err_out),
			POLYBENCH_ARRAY(delU),
			POLYBENCH_ARRAY(delW),
			POLYBENCH_ARRAY(delV),
			POLYBENCH_ARRAY(delTA),
			POLYBENCH_ARRAY(delTB));
	*/
	/* Stop and print timer. */
	polybench_stop_instruments;
	polybench_print_instruments;

	/* Prevent dead-code elimination. All live-out data must be printed
	   by the function call in argument. */
	polybench_prevent_dce(print_array_fwd(nt, nq,  POLYBENCH_ARRAY(out_F)));
	polybench_prevent_dce(print_array_bwd(np, nq, ns,  POLYBENCH_ARRAY(delU), 
				POLYBENCH_ARRAY(delW), POLYBENCH_ARRAY(delV), POLYBENCH_ARRAY(delTA)));

	/* Be clean. */
	POLYBENCH_FREE_ARRAY(out_F);
	POLYBENCH_FREE_ARRAY(s_F);
	POLYBENCH_FREE_ARRAY(inp_F);
	POLYBENCH_FREE_ARRAY(U);
	POLYBENCH_FREE_ARRAY(W);
	POLYBENCH_FREE_ARRAY(V);
	POLYBENCH_FREE_ARRAY(delU);
	POLYBENCH_FREE_ARRAY(delW);
	POLYBENCH_FREE_ARRAY(delV);
	POLYBENCH_FREE_ARRAY(delTA);    
	POLYBENCH_FREE_ARRAY(delTB);  
	POLYBENCH_FREE_ARRAY(err_out);  

	return 0;
}
