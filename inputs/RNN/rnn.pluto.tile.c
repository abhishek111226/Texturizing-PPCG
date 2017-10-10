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
  int t1, t2, t3, t4, t5, t6, t7;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NS >= 1) && (_PB_NT >= 1)) {
  if (_PB_NP >= 1) {
    for (t2=0;t2<=floord(_PB_NT-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NS-1,32);t3++) {
        for (t4=0;t4<=floord(_PB_NP-1,32);t4++) {
          for (t5=32*t2;t5<=min(_PB_NT-1,32*t2+31);t5++) {
            for (t6=32*t3;t6<=min(_PB_NS-1,32*t3+31);t6++) {
              for (t7=32*t4;t7<=min(_PB_NP-1,32*t4+31);t7++) {
                s_F[t5][t6] += U[t6][t7] * inp_F[t5][t7];;
              }
            }
          }
        }
      }
    }
  }
  if (_PB_NT >= 2) {
    for (t2=0;t2<=floord(_PB_NS-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NT-1,32);t3++) {
        for (t4=32*t2;t4<=min(_PB_NS-1,32*t2+31);t4++) {
          for (t5=max(1,32*t3);t5<=min(_PB_NT-1,32*t3+31);t5++) {
            s_F[t5][s1] += W[s1][t4] * s_F[t5-1][t4];;
          }
        }
      }
    }
  }
  if (_PB_NQ >= 1) {
    for (t2=0;t2<=floord(_PB_NT-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NQ-1,32);t3++) {
        for (t4=0;t4<=floord(_PB_NS-1,32);t4++) {
          for (t5=32*t2;t5<=min(_PB_NT-1,32*t2+31);t5++) {
            for (t6=32*t3;t6<=min(_PB_NQ-1,32*t3+31);t6++) {
              for (t7=32*t4;t7<=min(_PB_NS-1,32*t4+31);t7++) {
                out_F[t5][t6] += V[t6][t7] * s_F[t5][t7];;
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
  int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NS >= 1) && (_PB_NT >= 2)) {
  for (t2=-_PB_NT+1;t2<=-1;t2++) {
    for (t4=0;t4<=floord(_PB_NQ-1,32);t4++) {
      for (t6=0;t6<=floord(_PB_NS-1,32);t6++) {
        for (t7=32*t4;t7<=min(_PB_NQ-1,32*t4+31);t7++) {
          lbv=32*t6;
          ubv=min(_PB_NS-1,32*t6+31);
#pragma ivdep
#pragma vector always
          for (t9=lbv;t9<=ubv;t9++) {
            delV[t7][t9] = err_out[-t2][t7] * s_F[-t2][t9];;
          }
        }
      }
    }
    for (t4=0;t4<=floord(_PB_NS-1,32);t4++) {
      lbv=32*t4;
      ubv=min(_PB_NS-1,32*t4+31);
#pragma ivdep
#pragma vector always
      for (t7=lbv;t7<=ubv;t7++) {
        delTA[t7] = 0;;
      }
      for (t6=0;t6<=floord(_PB_NQ-1,32);t6++) {
        for (t7=32*t4;t7<=min(_PB_NS-1,32*t4+31);t7++) {
          for (t9=32*t6;t9<=min(_PB_NQ-1,32*t6+31);t9++) {
            delTA[t7] += V[t9][t7] * err_out[-t2][t9];;
          }
        }
      }
    }
    for (t4=t2-1;t4<=min(-1,t2+bptt_trunc-1);t4++) {
      for (t6=0;t6<=floord(_PB_NS-1,32);t6++) {
        for (t8=0;t8<=floord(_PB_NS-1,32);t8++) {
          for (t9=32*t6;t9<=min(_PB_NS-1,32*t6+31);t9++) {
            lbv=32*t8;
            ubv=min(_PB_NS-1,32*t8+31);
#pragma ivdep
#pragma vector always
            for (t11=lbv;t11<=ubv;t11++) {
              delW[t9][t11] += delTA[t9] * s_F[-t4 - 1][t11];;
            }
          }
        }
      }
      if (_PB_NP >= 1) {
        for (t6=0;t6<=floord(_PB_NS-1,32);t6++) {
          for (t8=0;t8<=floord(_PB_NP-1,32);t8++) {
            for (t9=32*t6;t9<=min(_PB_NS-1,32*t6+31);t9++) {
              lbv=32*t8;
              ubv=min(_PB_NP-1,32*t8+31);
#pragma ivdep
#pragma vector always
              for (t11=lbv;t11<=ubv;t11++) {
                delU[t9][t11] += delTA[t9] * inp_F[-t4][t11];;
              }
            }
          }
        }
      }
      for (t6=0;t6<=floord(_PB_NS-1,32);t6++) {
        lbv=32*t6;
        ubv=min(_PB_NS-1,32*t6+31);
#pragma ivdep
#pragma vector always
        for (t9=lbv;t9<=ubv;t9++) {
          delTB[t9] = 0;;
        }
        for (t8=0;t8<=floord(_PB_NS-1,32);t8++) {
          for (t9=32*t6;t9<=min(_PB_NS-1,32*t6+31);t9++) {
            for (t11=32*t8;t11<=min(_PB_NS-1,32*t8+31);t11++) {
              delTB[t9] += delTA[t11] * W[t11][t9];;
            }
          }
        }
      }
      for (t6=0;t6<=floord(_PB_NS-1,32);t6++) {
        lbv=32*t6;
        ubv=min(_PB_NS-1,32*t6+31);
#pragma ivdep
#pragma vector always
        for (t7=lbv;t7<=ubv;t7++) {
          delTA[t7] = delTB[t7];;
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

	rnn_backward(nt, np, ns, nq, bptt_trunc,  
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
