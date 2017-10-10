//#include <stdio.h>
int A[4][4];
int B[4][4];
int C[4][4];
int main()
{
 int i,j=0,k; 	
 #pragma scop
 #pragma texture ( A, B)

for(k=1;k< M; k++)
	for(p=0;p< NZ; p++)
		x[k][row[p]] += a[p]* x[k-1][col [p]];
 #pragma endscop
  return 0;
}

