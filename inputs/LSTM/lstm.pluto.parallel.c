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
if (_PB_T >= 1) {
  if ((_PB_P >= 1) && (_PB_S >= ns+1) && (ns >= 1)) {
    lbp=0;
    ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        g[t2] += U_g[t2][t9] * inp_F[0][t9];;
      }
      o[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        o[t2] += U_o[t2][t9] * inp_F[0][t9];;
      }
      f[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        f[t2] += U_f[t2][t9] * inp_F[0][t9];;
      }
      i[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        i[t2] += U_i[t2][t9] * inp_F[0][t9];;
      }
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
    lbp=ns;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        g[t2] += U_g[t2][t9] * inp_F[0][t9];;
      }
      o[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        o[t2] += U_o[t2][t9] * inp_F[0][t9];;
      }
      f[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        f[t2] += U_f[t2][t9] * inp_F[0][t9];;
      }
      i[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        i[t2] += U_i[t2][t9] * inp_F[0][t9];;
      }
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= 1) && (_PB_S <= ns-1)) {
    lbp=0;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        g[t2] += U_g[t2][t9] * inp_F[0][t9];;
      }
      o[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        o[t2] += U_o[t2][t9] * inp_F[0][t9];;
      }
      f[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        f[t2] += U_f[t2][t9] * inp_F[0][t9];;
      }
      i[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        i[t2] += U_i[t2][t9] * inp_F[0][t9];;
      }
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
    lbp=_PB_S;
    ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= 1) && (_PB_S == ns)) {
    lbp=0;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        g[t2] += U_g[t2][t9] * inp_F[0][t9];;
      }
      o[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        o[t2] += U_o[t2][t9] * inp_F[0][t9];;
      }
      f[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        f[t2] += U_f[t2][t9] * inp_F[0][t9];;
      }
      i[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        i[t2] += U_i[t2][t9] * inp_F[0][t9];;
      }
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= 1) && (ns <= 0)) {
    lbp=0;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        g[t2] += U_g[t2][t9] * inp_F[0][t9];;
      }
      o[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        o[t2] += U_o[t2][t9] * inp_F[0][t9];;
      }
      f[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        f[t2] += U_f[t2][t9] * inp_F[0][t9];;
      }
      i[t2] = 0.0;;
      for (t9=0;t9<=_PB_P-1;t9++) {
        i[t2] += U_i[t2][t9] * inp_F[0][t9];;
      }
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= ns+1) && (ns >= 1)) {
    lbp=0;
    ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      o[t2] = 0.0;;
      f[t2] = 0.0;;
      i[t2] = 0.0;;
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
    lbp=ns;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      o[t2] = 0.0;;
      f[t2] = 0.0;;
      i[t2] = 0.0;;
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= 1) && (_PB_S <= ns-1)) {
    lbp=0;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      o[t2] = 0.0;;
      f[t2] = 0.0;;
      i[t2] = 0.0;;
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
    lbp=_PB_S;
    ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= 1) && (_PB_S == ns)) {
    lbp=0;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      o[t2] = 0.0;;
      f[t2] = 0.0;;
      i[t2] = 0.0;;
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= 1) && (ns <= 0)) {
    lbp=0;
    ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      g[t2] = 0.0;;
      o[t2] = 0.0;;
      f[t2] = 0.0;;
      i[t2] = 0.0;;
    }
  }
  if ((_PB_S <= 0) && (ns >= 1)) {
    lbp=0;
    ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
    for (t2=lbp;t2<=ubp;t2++) {
      s_F[0][t2] = c_F[0][t2] * o[t2];;
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= ns+1) && (ns >= 1)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          g[t2] += U_g[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          o[t2] += U_o[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          f[t2] += U_f[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          i[t2] += U_i[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
      lbp=ns;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          g[t2] += U_g[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          o[t2] += U_o[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          f[t2] += U_f[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          i[t2] += U_i[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
      }
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= 1) && (_PB_S <= ns-1)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          g[t2] += U_g[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          o[t2] += U_o[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          f[t2] += U_f[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          i[t2] += U_i[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
      lbp=_PB_S;
      ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= 1) && (_PB_S == ns)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          g[t2] += U_g[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          o[t2] += U_o[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          f[t2] += U_f[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          i[t2] += U_i[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
    }
  }
  if ((_PB_P >= 1) && (_PB_S >= 1) && (ns <= 0)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          g[t2] += U_g[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          o[t2] += U_o[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          f[t2] += U_f[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_P-1;t9++) {
          i[t2] += U_i[t2][t9] * inp_F[t1][t9];;
        }
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
      }
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= ns+1) && (ns >= 1)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
      lbp=ns;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
      }
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= 1) && (_PB_S <= ns-1)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
      lbp=_PB_S;
      ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= 1) && (_PB_S == ns)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
    }
  }
  if ((_PB_P <= 0) && (_PB_S >= 1) && (ns <= 0)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        g[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          g[t2] += W_g[t2][t9] * s_F[t1-1][t9];;
        }
        o[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          o[t2] += W_o[t2][t9] * s_F[t1-1][t9];;
        }
        f[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          f[t2] += W_f[t2][t9] * s_F[t1-1][t9];;
        }
        i[t2] = 0.0;;
        for (t9=0;t9<=_PB_S-1;t9++) {
          i[t2] += W_i[t2][t9] * s_F[t1-1][t9];;
        }
      }
    }
  }
  if ((_PB_S <= 0) && (ns >= 1)) {
    for (t1=1;t1<=_PB_T-1;t1++) {
      lbp=0;
      ubp=ns-1;
#pragma omp parallel for private(lbv,ubv,t3,t4,t5,t6,t7,t8,t9)
      for (t2=lbp;t2<=ubp;t2++) {
        c_F[t1][t2] = c_F[t1-1][t2] * f[t2] + g[t2] * i[t2];;
        s_F[t1][t2] = c_F[t1][t2] * o[t2];;
      }
    }
  }
}
/* End of CLooG code */
}

	static
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
if ((_PB_S >= 1) && (_PB_T >= 2)) {
  if (_PB_P >= 1) {
    for (t2=-_PB_T+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_o[t4] = c_F[-t2][t4] * del_S[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_C[-t2][t4] += o[t4] * del_S[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_f[t4] = c_F[-t2-1][t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_C[-t2-1][t4] = f[t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_i[t4] = g[t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_g[t4] = i[t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_P-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Ui[t4][t6] = inp_F[-t2][t6] * i[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_P-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Uf[t4][t6] = inp_F[-t2][t6] * f[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_P-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Uo[t4][t6] = inp_F[-t2][t6] * o[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_P-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Ug[t4][t6] = inp_F[-t2][t6] * g[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wi[t4][t6] = s_F[-t2-1][t6] * i[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wf[t4][t6] = s_F[-t2-1][t6] * f[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wo[t4][t6] = s_F[-t2-1][t6] * o[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wg[t4][t6] = s_F[-t2-1][t6] * g[t4];;
        }
      }
      for (t4=0;t4<=2*_PB_S-2;t4++) {
        lbp=max(0,t4-_PB_S+1);
        ubp=min(t4,_PB_S-1);
#pragma omp parallel for private(lbv,ubv,t7)
        for (t6=lbp;t6<=ubp;t6++) {
          del_S[-t2-1][t6] = W_i[t6][(t4-t6)] * del_i[(t4-t6)] + W_f[t6][(t4-t6)] * del_f[(t4-t6)] + W_o[t6][(t4-t6)] * del_o[(t4-t6)] + W_g[t6][(t4-t6)] * del_g[(t4-t6)];;
        }
      }
    }
  }
  if (_PB_P <= 0) {
    for (t2=-_PB_T+1;t2<=-1;t2++) {
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_o[t4] = c_F[-t2][t4] * del_S[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_C[-t2][t4] += o[t4] * del_S[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_f[t4] = c_F[-t2-1][t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_C[-t2-1][t4] = f[t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_i[t4] = g[t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        del_g[t4] = i[t4] * del_C[-t2][t4];;
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wi[t4][t6] = s_F[-t2-1][t6] * i[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wf[t4][t6] = s_F[-t2-1][t6] * f[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wo[t4][t6] = s_F[-t2-1][t6] * o[t4];;
        }
      }
      lbp=0;
      ubp=_PB_S-1;
#pragma omp parallel for private(lbv,ubv,t5,t6,t7)
      for (t4=lbp;t4<=ubp;t4++) {
        lbv=0;
        ubv=_PB_S-1;
#pragma ivdep
#pragma vector always
        for (t6=lbv;t6<=ubv;t6++) {
          del_Wg[t4][t6] = s_F[-t2-1][t6] * g[t4];;
        }
      }
      for (t4=0;t4<=2*_PB_S-2;t4++) {
        lbp=max(0,t4-_PB_S+1);
        ubp=min(t4,_PB_S-1);
#pragma omp parallel for private(lbv,ubv,t7)
        for (t6=lbp;t6<=ubp;t6++) {
          del_S[-t2-1][t6] = W_i[t6][(t4-t6)] * del_i[(t4-t6)] + W_f[t6][(t4-t6)] * del_f[(t4-t6)] + W_o[t6][(t4-t6)] * del_o[(t4-t6)] + W_g[t6][(t4-t6)] * del_g[(t4-t6)];;
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

	lstm_backward(nt, np, ns, nq,
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
	   POLYBENCH_ARRAY(del_g));

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
