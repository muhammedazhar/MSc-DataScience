#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>


int main (int argc, char *argv[])
{
	int i, tid;
	double start, end;
	int N=16;
	int THREADS=2;
	
	if (argc==3){
		THREADS=atoi(argv[1]);
		N=atoi(argv[2]);
	}
	else{
	printf("Number of threads and N defaulting to 2 and 16 \n");
	}
	//Set the number of threads that we are going to use
	omp_set_num_threads(THREADS);
	start = omp_get_wtime();
	//start of parallel section
	#pragma omp parallel private(i)
	{
		#pragma omp for schedule(dynamic)
		for (i = 0; i < N; i++) {
			/* wait for 0.5*i seconds.
			 * This emulates a variable "processing time" for each thread.
			 * This "processing time" is based on the iteration number
			 * that is assigned to the thread.
			*/ 
			sleep(0.5*i);
			tid = omp_get_thread_num( );
			printf("Thread %d has completed iteration %d.\n", tid, i);
		}


	}// end of parallel section
	// all threads done

	end = omp_get_wtime();
	printf("Work took %f seconds\n", end - start);
	printf("All done!\n");
	return 0;
}
