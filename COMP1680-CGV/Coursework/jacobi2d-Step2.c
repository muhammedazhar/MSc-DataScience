#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>     // Added OpenMP library for parallel execution

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s m n tolerance num_threads\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    double tol = atof(argv[3]);
    int num_threads = atoi(argv[4]);
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Output the number of threads being used
    printf("Number of threads being used: %d\n", num_threads);
    
    // Start timing
    double start_time, end_time;
    start_time = omp_get_wtime();
    
    // Dynamically allocate 2D arrays
    double **t = (double **)malloc((m + 2) * sizeof(double *));
    double **tnew = (double **)malloc((m + 2) * sizeof(double *));
    
    for (int i = 0; i < m + 2; i++) {
        t[i] = (double *)malloc((n + 2) * sizeof(double));
        tnew[i] = (double *)malloc((n + 2) * sizeof(double));
    }

    printf("%d %d %lf %d\n", m, n, tol, num_threads);

    // Initialize temperature array
    #pragma omp parallel for collapse(2)
    for (int i = 0; i <= m + 1; i++) {
        for (int j = 0; j <= n + 1; j++) {
            t[i][j] = 30.0;
        }
    }

    // Fix boundary conditions to match Step 1 requirements
    #pragma omp parallel for
    for (int i = 1; i <= m; i++) {
        t[i][0] = 47.0;        // Left boundary set to 47°C
        t[i][n + 1] = 100.0;   // Right boundary set to 100°C
    }
    #pragma omp parallel for
    for (int j = 1; j <= n; j++) {
        t[0][j] = 15.0;        // Top boundary set to 15°C
        t[m + 1][j] = 60.0;    // Bottom boundary set to 60°C
    }

    // Set corner values as the average of their adjacent boundary values
    t[0][0] = (15.0 + 47.0) / 2.0;          // Top-left corner
    t[0][n + 1] = (15.0 + 100.0) / 2.0;     // Top-right corner
    t[m + 1][0] = (60.0 + 47.0) / 2.0;      // Bottom-left corner
    t[m + 1][n + 1] = (60.0 + 100.0) / 2.0; // Bottom-right corner

    // Main loop
    int iter = 0;
    double difmax = 1000000.0;
    
    while (difmax > tol) {
        iter++;
        difmax = 0.0;

        // Update temperature for next iteration in parallel
        #pragma omp parallel for collapse(2) shared(t, tnew)
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                tnew[i][j] = (t[i-1][j] + t[i+1][j] + t[i][j-1] + t[i][j+1]) / 4.0;
            }
        }

        // Calculate maximum difference in parallel
        #pragma omp parallel for collapse(2) reduction(max: difmax)
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                double diff = fabs(tnew[i][j] - t[i][j]);
                difmax = (diff > difmax) ? diff : difmax;
                t[i][j] = tnew[i][j];
            }
        }
    }

    // End timing
    end_time = omp_get_wtime();
    double exec_time = end_time - start_time;

    // Print results
    printf("iter = %d  difmax = %9.11lf\n", iter, difmax);
    // Uncomment the following lines for smaller problem sizes to print the grid
    // for (int i = 0; i <= m + 1; i++) {
    //     for (int j = 0; j <= n + 1; j++) {
    //         printf("%3.5lf ", t[i][j]);
    //     }
    //     printf("\n");
    // }
    printf("Execution time: %f seconds\n", exec_time);

    // Free allocated memory
    for (int i = 0; i < m + 2; i++) {
        free(t[i]);
        free(tnew[i]);
    }
    free(t);
    free(tnew);

    return 0;
}
