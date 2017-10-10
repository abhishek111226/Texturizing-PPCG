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
#pragma scop

	for (t = 0; t < _PB_NT; t++)
	{    
		for(s1 = 0; s1 < _PB_NS; s1++)
			for(p = 0; p < _PB_NP; p++)
				s_F[t][s1] += U[s1][p] * inp_F[t][p];

		if(t > 0)
			for(s2 = 0; s2 < _PB_NS; s2++)
				s_F[t][s1] += W[s1][s2] * s_F[t-1][s2];

		for(q = 0; q < _PB_NQ; q++)
			for(s1 = 0; s1 < _PB_NS; s1++)
				out_F[t][q] += V[q][s1] * s_F[t][s1];
	}
#pragma endscop
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
#pragma scop

	for (t = _PB_NT - 1; t > 0; t--)
	{  
		for(q = 0; q < _PB_NQ; q++)
			for(s = 0; s < _PB_NS; s++)
				delV[q][s] = err_out[t][q] * s_F[t][s];

		for(s = 0; s < _PB_NS; s++)
		{
			delTA[s] = 0;
			for(q = 0; q < _PB_NQ; q++)
				delTA[s] += V[q][s] * err_out[t][q];
		}
		for(step=t+1; step > MAX(0, t - bptt_trunc); step--)
		{
			if(step > 0)
				for(r = 0; r < _PB_NS; r++)
					for(s = 0; s < _PB_NS; s++)
						delW[r][s] += delTA[r] * s_F[step - 1][s];

			for(s = 0; s < _PB_NS; s++)
				for(p = 0; p < _PB_NP; p++)
					delU[s][p] += delTA[s] * inp_F[step][p];

			for(r = 0; r < _PB_NS; r++)
			{
				delTB[r] = 0;
				for(s = 0; s < _PB_NS; s++)
					delTB[r] += delTA[s] * W[s][r];  
			}

			for(r = 0; r < _PB_NS; r++)
				delTA[r] = delTB[r];
		}
	}	
#pragma endscop

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
