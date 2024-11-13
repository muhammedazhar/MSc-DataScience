#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


int main(int argc, char** argv)
{
	//The variables for the start and stop timer
	struct timeval startTime, stopTime;
    int n=1000;
	int THREADS=2;
    long a[1000000];
	long long sum;
	long long sum_priv;
	int i;
	 // This variable will hold the total time
    long totalTime;

    // Start timer: get current time and store it in variable startTime
    gettimeofday(&startTime, NULL);
	//We use atoi() to read an integer command line argument
    if (argc==3){
		THREADS=atoi(argv[1]);
		n=atoi(argv[2]);
	}
	else{
	printf("Number of threads and squares expected defaulting to 2 threads and 1000 squares\n");
	}
	
    sum=0; 
	#pragma omp parallel private(i,sum_priv)
	 {
		sum_priv=0;
		#pragma omp for schedule(static)
		for(i=0; i<n; i++){
			a[i] = (i+1)*(i+1);	
			printf("%d , %ld\n", i+1, a[i]);
			sum_priv = sum_priv+a[i];
		}
		#pragma omp critical
		sum = sum+sum_priv;
	}
	// Stop timer: get current time and store it in variable stopTime
    gettimeofday(&stopTime, NULL);

    // Calculate total time by subracting the startTime from the stopTime (result is in microseconds)
    totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) - (startTime.tv_sec * 1000000 + startTime.tv_usec);
	
	printf("The Sum of %d squares is %lld\n", n, sum);
	    // Print the totalTime as a long integer (%ld)
    printf("%ld\n", totalTime);
    
	return 0;
}
//note this code will have unexpected results due to variable clash on the sum variable