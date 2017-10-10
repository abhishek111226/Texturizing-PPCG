//#include <stdio.h>
int A[4][4];
int B[4][4];
int C[4][4];
int main()
{
 int i,j=0,k; 	
   int temp;
   for (int c0 = 1; c0 <= 3; c0 += 1) {
     temp = B[c0];
     C[c0][0] = A[temp][0];
 }
  return 0;
}

