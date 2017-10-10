int tex_array[10][10];
//int cangle[360];
int main()
{	
int t, i, j,loop,sum1,sum2,a;
 for(int j=0;j<10;j++)
 for(int i=0;i<10;i++)
 {

	tex_array[i][j]=i+j;
 }
#pragma scop
 for(int j=1;j<10;j++)
  for(int i=1;i<10;i++)
  {	

	if(tex_array[i][j]==0)
	{
		//tex_array[i]=i*2;
		tex_array[i][j] = i*2;
		//sum1+=tex_array[i];
	}
	else if(i%3==2)
	{
		//tex_array[i]=tex_array[i-5];
		tex_array[i][j] = i*2+1;
		//sum1+=tex_array[i];
	}
	else
	{
		tex_array[i][j] = i*2+2;
	}
	
  	//for(int loop=0;loop<360;loop++)
          //       array[i]= array [i] + cangle [loop] ;
	 //tex_array[i]=tex_array[i-1] ;			
  }	
#pragma endscop
 for(int j=0;j<10;j++)
 for(int i=0;i<10;i++)
 {
	sum1+=tex_array[i][j];
 }
  return 0;
}
