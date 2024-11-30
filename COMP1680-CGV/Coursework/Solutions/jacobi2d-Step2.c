/*
 * COMP1680-CGV Coursework Step 2 - OpenMP Implementation
 * ------------------------------------------------------
 * OpenMP Parallel Programming for Jacobi 2D (20x20) grid problem
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>    // Added OpenMP library for parallel execution

// Boundary temperatures for the 2D grid
#define TOP_TEMP 15.0    // Top boundary temperature
#define BOTTOM_TEMP 60.0 // Bottom boundary temperature
#define LEFT_TEMP 47.0   // Left boundary temperature
#define RIGHT_TEMP 100.0 // Right boundary temperature

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc != 4) {
        printf("Usage: %s m n tolerance\n", argv[0]);
        return 1;
    }

    // Get grid dimensions and tolerance from command line
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    double tol = atof(argv[3]);
    int i, j;
    double diff;
    
    // Start timing the execution
    double start_time = omp_get_wtime();

    // Allocate memory for temperature grids
    // Using contiguous memory layout for better cache performance
    double *t_data = (double *)malloc((m + 2) * (n + 2) * sizeof(double));
    double *tnew_data = (double *)malloc((m + 2) * (n + 2) * sizeof(double));
    double **t = (double **)malloc((m + 2) * sizeof(double *));
    double **tnew = (double **)malloc((m + 2) * sizeof(double *));

    // Setup 2D array pointers
    for (i = 0; i < m + 2; i++) {
        t[i] = &t_data[i * (n + 2)];
        tnew[i] = &tnew_data[i * (n + 2)];
    }

    // Initialize parallel region for setting up initial conditions
    #pragma omp parallel
    {
        // Initialize interior points to 30.0
        #pragma omp for collapse(2) schedule(static) nowait
        for (i = 0; i <= m + 1; i++) {
            for (j = 0; j <= n + 1; j++) {
                t[i][j] = 30.0;
                tnew[i][j] = 30.0;
            }
        }

        // Set left and right boundary temperatures
        #pragma omp for schedule(static) nowait
        for (i = 1; i <= m; i++) {
            t[i][0] = LEFT_TEMP;
            t[i][n + 1] = RIGHT_TEMP;
        }

        // Set top and bottom boundary temperatures
        #pragma omp for schedule(static)
        for (j = 1; j <= n; j++) {
            t[0][j] = TOP_TEMP;
            t[m + 1][j] = BOTTOM_TEMP;
        }
    }

    // Set corner temperatures as average of adjacent boundaries
    t[0][0] = (TOP_TEMP + LEFT_TEMP) / 2.0;
    t[0][n + 1] = (TOP_TEMP + RIGHT_TEMP) / 2.0;
    t[m + 1][0] = (BOTTOM_TEMP + LEFT_TEMP) / 2.0;
    t[m + 1][n + 1] = (BOTTOM_TEMP + RIGHT_TEMP) / 2.0;

    // Main iteration loop
    int iter = 0;
    double difmax = 1000000.0;

    while (difmax > tol) {
        iter++;
        difmax = 0.0;

        // Using OpenMP to parallelize the following array operations
        #pragma omp parallel
        {
            // Parallelizes the loop with OpenMP, using static scheduling and no wait clause
            #pragma omp for private(j) schedule(static) nowait
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= n; j++) {
                    tnew[i][j] = (t[i-1][j] + t[i+1][j] + t[i][j-1] + t[i][j+1]) / 4.0;
                }
            }

            // Parallelizes the loop with OpenMP, using private variables j and diff, and a max reduction on difmax
            #pragma omp for private(j, diff) reduction(max:difmax)
            for (i = 1; i <= m; i++) {
                for (j = 1; j <= n; j++) {
                    diff = fabs(tnew[i][j] - t[i][j]);
                    difmax = (diff > difmax) ? diff : difmax;
                    t[i][j] = tnew[i][j];
                }
            }
        }
    }

    // Calculate and print execution time and results
    double exec_time = omp_get_wtime() - start_time;

    printf("iter = %d  difmax = %9.11lf\n", iter, difmax);
    // Print grid values for small grids (20x20 or smaller)
    if (m <= 20 && n <= 20) {
        for (i = 0; i <= m + 1; i++) {
            for (j = 0; j <= n + 1; j++) {
                printf("%3.5lf ", t[i][j]);
            }
            printf("\n");
        }
    }
    printf("Execution time: %f seconds\n", exec_time);

    // Free allocated memory
    free(t_data);
    free(tnew_data);
    free(t);
    free(tnew);

    return 0;
}
