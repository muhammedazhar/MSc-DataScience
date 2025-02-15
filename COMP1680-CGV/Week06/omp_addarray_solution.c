/******************************************************************************
* FILE: omp_workshare1.c
* DESCRIPTION:
*   OpenMP Example - Loop Work-sharing - C/C++ Version
*   In this example, the iterations of a loop are scheduled dynamically
*   across the team of threads.  A thread will perform CHUNK iterations
*   at a time before being scheduled for the next CHUNK of work.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define CHUNKSIZE   10
#define N       100
#define THREADS 4

int main (int argc, char *argv[])
{
	int nthreads, tid, i, chunk;
	float a[N], b[N], c[N];
	double start, end;
	
	start = omp_get_wtime();
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
	end = omp_get_wtime();
	printf("Work took %f seconds\n", end - start);

}
