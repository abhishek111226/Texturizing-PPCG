int tex_array[2];
//int cangle[360];
int main()
{	
int t, i, j,loop,sum;
 for(int i=0;i<2;i++)
 {
	tex_array[i]=i;
 }
#pragma scop
  for(int i=0;i<2;i++)
  {		
  	//for(int loop=0;loop<360;loop++)
          //       array[i]= array [i] + cangle [loop] ;
	int a;
	a = tex_array[i];			
  }	
#pragma endscop
  return 0;
}
