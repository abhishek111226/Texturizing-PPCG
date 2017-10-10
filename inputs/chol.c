//#include <stdio.h>
int A[3][3];
int B[3][3];
int C[3];
int main()
{
  int i,j,k;
#pragma scop
  for (i = 0; i < 3; i++) {
     //j<i
     for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }
     // i==j case
     for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = A[i][i];
  }
#pragma endscop
printf("hello");
return 0;
}
