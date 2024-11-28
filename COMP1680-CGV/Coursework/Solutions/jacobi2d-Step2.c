#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TOP_TEMP 15.0    // Top boundary temperature
#define BOTTOM_TEMP 60.0 // Bottom boundary temperature
#define LEFT_TEMP 47.0   // Left boundary temperature
#define RIGHT_TEMP 100.0 // Right boundary temperature

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage: %s m n tolerance\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    double tol = atof(argv[3]);

    double start_time = omp_get_wtime();

    // Optimized memory allocation for better cache utilization
    double *t_data = (double *)malloc((m + 2) * (n + 2) * sizeof(double));
    double *tnew_data = (double *)malloc((m + 2) * (n + 2) * sizeof(double));
    double **t = (double **)malloc((m + 2) * sizeof(double *));
    double **tnew = (double **)malloc((m + 2) * sizeof(double *));

    for (int i = 0; i < m + 2; i++)
    {
        t[i] = &t_data[i * (n + 2)];
        tnew[i] = &tnew_data[i * (n + 2)];
    }

    printf("%d %d %lf\n", m, n, tol);

// Combined initialization and boundary conditions in one parallel region
#pragma omp parallel
    {
// Initialize temperature array
#pragma omp for collapse(2) schedule(static) nowait
        for (int i = 0; i <= m + 1; i++)
        {
            for (int j = 0; j <= n + 1; j++)
            {
                t[i][j] = 30.0;
                tnew[i][j] = 30.0;
            }
        }

// Set boundary conditions
#pragma omp for schedule(static) nowait
        for (int i = 1; i <= m; i++)
        {
            t[i][0] = LEFT_TEMP;      // Left boundary
            t[i][n + 1] = RIGHT_TEMP; // Right boundary
        }

#pragma omp for schedule(static)
        for (int j = 1; j <= n; j++)
        {
            t[0][j] = TOP_TEMP;        // Top boundary
            t[m + 1][j] = BOTTOM_TEMP; // Bottom boundary
        }
    }

    // Set corner values
    t[0][0] = (TOP_TEMP + LEFT_TEMP) / 2.0;
    t[0][n + 1] = (TOP_TEMP + RIGHT_TEMP) / 2.0;
    t[m + 1][0] = (BOTTOM_TEMP + LEFT_TEMP) / 2.0;
    t[m + 1][n + 1] = (BOTTOM_TEMP + RIGHT_TEMP) / 2.0;

    int iter = 0;
    double difmax = 1000000.0;

    // Main computation loop
    while (difmax > tol)
    {
        iter++;
        difmax = 0.0;

#pragma omp parallel
        {
// Temperature update with nowait as next loop needs all values
#pragma omp for private(j, diff) shared(t, tnew) reduction(max : difmax) nowait
            for (int i = 1; i <= m; i++)
            {
                for (int j = 1; j <= n; j++)
                {
                    tnew[i][j] = (t[i - 1][j] + t[i + 1][j] + t[i][j - 1] + t[i][j + 1]) / 4.0;
                }
            }

// Implicit barrier before this loop ensures all tnew values are ready
#pragma omp for private(j) shared(t, tnew)
            for (int i = 1; i <= m; i++)
            {
                for (int j = 1; j <= n; j++)
                {
                    double diff = fabs(tnew[i][j] - t[i][j]);
                    difmax = (diff > difmax) ? diff : difmax;
                    t[i][j] = tnew[i][j];
                }
            }
        }
    }

    double exec_time = omp_get_wtime() - start_time;

    printf("iter = %d  difmax = %9.11lf\n", iter, difmax);
    if (m <= 20 && n <= 20)
    {
        for (int i = 0; i <= m + 1; i++)
        {
            for (int j = 0; j <= n + 1; j++)
            {
                printf("%3.5lf ", t[i][j]);
            }
            printf("\n");
        }
    }
    printf("Execution time: %f seconds\n", exec_time);

    // Cleanup
    free(t_data);
    free(tnew_data);
    free(t);
    free(tnew);

    return 0;
}