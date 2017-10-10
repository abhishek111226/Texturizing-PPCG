//#include <stdio.h>
int A[300][300][300];
int B[300][300][300];
int main()
{
 int i,j,k; 	
 #pragma scop
 for(i=1;i<300;i++)
 {
 	for(j=0;j<300;j++)
 	{
		 	for(k=0;k<300;k++)
		 	{
				B[i][j][k]=A[i-1][j][k];
			}
	}
 }
 #pragma endscop
  return 0;
}

