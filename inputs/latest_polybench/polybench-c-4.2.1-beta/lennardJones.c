/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "lennardJones.h"
/* Default data type is double, default size is N=1024. */
//#include "template-for-new-benchmark.h"


/* Array initialization. */
static
void init_array(int n, DATA_TYPE POLYBENCH_2D(x,N,3,n,3), DATA_TYPE POLYBENCH_2D(f,N,3,n,3), int POLYBENCH_2D(neigh,N,n,N+1,n+1))
{
  int i, j;
  double rho = 0.8442;
  double delta = 0.2662;	
  double alat = pow((4.0 / rho), (1.0 / 3.0));	
  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++)
    {
	  x[i][j] =  0.5 * alat * j + delta/1000*(rand()%1000-500);
	  f[i][j] = 0;
    }
    //for (j = 0; j < n+1; j++)
    //{
    //	  neigh[i][j] = 0;
   // }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, DATA_TYPE POLYBENCH_2D(f,N,3,n,3))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < 3; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, f[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_template(int n, DATA_TYPE POLYBENCH_2D(x,N,3,n,3), DATA_TYPE POLYBENCH_2D(f,N,3,n,3), int POLYBENCH_2D(neigh,N,n,N+1,n+1))
{
  int i, j;
  double cutforcesq=FORCE_CUTSEQ;            //force cutoff
  double epsilon=1.0;               //Potential parameter
  double sigma6=1.0; //Potential parameter
  double fix=0;
  double fiy=0;
  double fiz=0;


  double rm = 1.5 * FORCE_CUTSEQ ;   
  int i,j;
  for (i = 0; i < _PB_N; i++)
  {
	fix=0;
	fiy=0;
	fiz=0;
	const double xtmp = x[i][0];
	const double ytmp = x[i][1];
	const double ztmp = x[i][2];
	for (j = 0; j < _PB_N; j++)
	{
		if(i!=j)
		{
			const double delx = xtmp - x[j][0];
			const double dely = ytmp - x[j][1];
			const double delz = ztmp - x[j][2];
			const double rsq = delx * delx + dely * dely + delz * delz;
			if(rsq < rm) 
			{







#pragma scop
  for (i = 0; i < _PB_N; i++)
  {
	
	int neighbours = neigh[i][0];  
	for (j = 0; j < _PB_N; j++)
	{
			//neigh[i][j+1]
			//neigh[i][j+1]
			const double delx = xtmp - x[MIN(neigh[i][j+1], n)][0];
			const double dely = ytmp - x[1][1];
			const double delz = ztmp - x[1][2];
			const double rsq = delx * delx + dely * dely + delz * delz; 
			if(rsq < cutforcesq) 
			{
				//printf("\nEntered CS");
				const double sr2 = 1.0 / rsq;
				const double sr6 = sr2 * sr2 * sr2  * sigma6;
				const double force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
				fix += delx * force;
				fiy += dely * force;
				fiz += delz * force;
			}
	}
    f[i][0] += fix;
    f[i][1] += fiy;
    f[i][2] += fiz;
  }	 
#pragma endscop
}		



int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(x,DATA_TYPE,N,3,n,3);
  POLYBENCH_2D_ARRAY_DECL(f,DATA_TYPE,N,3,n,3);
  POLYBENCH_2D_ARRAY_DECL(neigh,int,N,N+1,n,n+1);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(f), POLYBENCH_ARRAY(neigh));


  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  setup_neighbours(n, POLYBENCH_ARRAY(x),POLYBENCH_ARRAY(neigh)); 	
  kernel_template (n, POLYBENCH_ARRAY(x),POLYBENCH_ARRAY(f), POLYBENCH_ARRAY(neigh));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n,  POLYBENCH_ARRAY(f)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(f);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(neigh);	
  return 0;
}
