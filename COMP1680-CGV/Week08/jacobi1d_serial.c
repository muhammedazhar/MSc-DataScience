#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int max_size=100000;

int main(int argc, char *argv[])
{
double told[max_size], t[max_size], tol=0.0001, diff, difmax;
int i, iter, n=1000;

if (argc==3){
		n=atoi(argv[1]);
		tol=atof(argv[2]);
	}
else{
	printf("Number of cells and tolerance expected defaulting to 1000 and 0.0001 \n");
	}

// initialise temperature array
for (i=1; i <= n; i++) {
   told[i] = 20.0;
   }
// fix end points as cold and hot
told[0] = 0.0;
told[n+1] = 100.0;

iter = 0;
difmax = 1000000.0;

while (difmax > tol) {

iter=iter+1;
// update temperature for next iteration
for (i=1; i <= n; i++) {
   t[i] = (told[i-1]+told[i+1])/2.0;
   }
// work out maximum difference between old and new temperatures
difmax=0.0;
for (i=1; i <= n; i++) {
   diff = fabs(t[i]-told[i]);
   if (diff > difmax) {
      difmax = diff;
      }
   told[i] = t[i];
   }

}//while (difmax > tol)



   for (i=1; i <= n; i++) {
      printf("told[%d] = %-5.7lf  \n", i, t[i]);
      }
   printf("iterations = %d  maximum difference = %f  \n", iter, difmax);
   
}

