/******************************************************************************
* FILE: omp_workshare1.c
* DESCRIPTION:
*   OpenMP Example - Loop Work-sharing - C/C++ Version
*   In this example, the iterations of a loop are scheduled dynamically
*   across the team of threads.  A thread will perform CHUNK iterations
*   at a time before being scheduled for the next CHUNK of work.
* AUTHOR: Blaise Barney  5/99
* MODIFIED BY: C Tonry 10/20
* LAST REVISED: 18/10/22
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
	int nthreads, tid, i, chunk;
	float a[10000], b[10000], c[10000];
	int CHUNKSIZE=10;
	int N=100;
	int THREADS=4;
	
	if (argc==4){
		THREADS=atoi(argv[1]);
		N=atoi(argv[2]);
		CHUNKSIZE=atoi(argv[3]);
	}
	else{
	printf("Number of threads, N and chunksize expected defaulting to 4, 100 and 10. \n");
	}
	/* Some initializations */
	for (i=0; i < N; i++)
		a[i] = b[i] = i * 1.0;
	chunk = CHUNKSIZE;
	omp_set_num_threads(THREADS);

	#pragma omp parallel shared(a,b,c,nthreads,chunk) private(i,tid)
	{
		tid = omp_get_thread_num();
		if (tid == 0) {
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads);
		}
		printf("Thread %d starting...\n",tid);

		#pragma omp for schedule(static,chunk)
		for (i=0; i<N; i++) {
			c[i] = a[i] + b[i];
			printf("Thread %d: c[%d]= %f\n",tid,i,c[i]);
		}

	}  /* end of parallel section */

}
