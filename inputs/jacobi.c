int tex_A[5];
int tex_B[5];
int tex_C[10];
int main()
{
 int i,t; 	
#pragma scop
  //for (t = 0; t < 2; t++)
    {
      for (i = 1; i < 4; i++)
      {
		tex_B[i] = tex_A[i]; 
		tex_C[i]= tex_B[i];
      }
    }
#pragma endscop
#pragma scop
  //for (t = 0; t < 2; t++)
    {
      for (i = 1; i < 4; i++)
      {
		tex_B[i] = tex_A[i]; 
		tex_C[i]=  tex_B[i];
      }
    }
#pragma endscop

  return 0;
}

