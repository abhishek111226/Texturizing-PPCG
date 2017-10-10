#include<stdio.h>
int A[4][4];
int B[4][4];
int main()
{	

	int t, i, j,tm,tj;
	#pragma scop
	  for (t = 0; t < 1; t++)
	    {
	      for (i = 1; i < 3; i++)	
		for (j = 1; j < 3; j++)	
		  B[i][j] = (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
	      for (i = 1; i < 3; i++)
		for (j = 1; j < 3; j++)
		  A[i][j] = (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);	
	    }
	#pragma endscop

  return 0;
}
