#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int max_size=400000;

int main(int argc, char *argv[]) {
    double told[max_size], t[max_size], tol=0.0001, diff=0.0001, difmax, priv_difmax;
    double tstart, tstop;
    int i, iter, n=1000, nthreads=2;
 
    
	if (argc==4){
		n=atoi(argv[1]);
		tol=atof(argv[2]);
		nthreads=atoi(argv[3]);
	}
	else{
	printf("Number of cells, tolerance and number of threads expected defaulting to 1000, 0.0001 and 2 \n");
	}
    
    
    
    /* define the number of threads to be used */
    omp_set_num_threads(nthreads);
    
    tstart = omp_get_wtime ();
    
    // initialise temperature array
    #pragma omp parallel for schedule(static) \
        default(shared) private(i)
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
        #pragma omp parallel for schedule(static) \
            default(shared) private(i)
        for (i=1; i <= n; i++) {
            t[i] = (told[i-1]+told[i+1])/2.0;
        }
        
        // work out maximum difference between old and new temperatures
        difmax = 0.0;
        
        #pragma omp parallel default(shared) private(i, diff, priv_difmax)
        {
            priv_difmax = 0.0;
            #pragma omp for schedule(static)
            for (i=1; i <= n; i++) {
                diff = fabs(t[i]-told[i]);
                if (diff > priv_difmax) {
                    priv_difmax = diff;
                }
                told[i] = t[i];
            }
            #pragma omp critical
            if (priv_difmax > difmax) {
                difmax = priv_difmax;
            }
        }
        
    }//while (difmax>tol)
    
    tstop = omp_get_wtime ();

    
    for (i=1; i <= n; i++) {
        printf("told[%d] = %-5.7lf  \n", i, t[i]);
    }
    printf("iterations = %d  maximum difference = %-5.7lf  \n", iter, difmax);
    
    
    printf("time taken is %4.3lf\n", (tstop-tstart));
}

