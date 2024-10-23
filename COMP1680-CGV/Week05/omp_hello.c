#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char** argv)
{
    int tid, nthreads;
    int THREADS = 2;
    
    if (argc==2){
        THREADS=atoi(argv[1]);
    }
    else{
    printf("Number of threads expected defaulting to 2. \n");
    }
    
    omp_set_num_threads(THREADS);
    
    // start of parallel section
    // Fork a team of threads with each thread having a private tid variable
    #pragma omp parallel private(tid)
    {
        tid=omp_get_thread_num();
        printf("Hello world from thread %d\n", tid);
        /* Only master thread does this */
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }


    }//end of parallel section
    // All threads join master thread and terminate


    return 0;
}  // end main()
