//#include<stdio.h>

int array[3][3];
int b[3][3];
int main()
{	
int t, i, j,loop,sum1,sum2,a,id;
 for (i = 1; i < 3; i++)
	for (j = 1; j < 3; j++)
	{
		b[i][j]=i+j+1;
	}
#pragma scop
#pragma texture (b)
  for (i = 1; i < 3; i++)
	for (j = 1; j < 3; j++)
	{
		array[i][j]=b[i][j];
	}
#pragma endscop
 for (i = 1; i < 3; i++)
	for (j = 1; j < 3; j++)
	{
		printf("%d ",array[i][j]);
	}
  return 0;
}
