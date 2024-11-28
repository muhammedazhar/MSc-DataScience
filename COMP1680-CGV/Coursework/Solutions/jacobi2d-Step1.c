/*
 * COMP1680-CGV Coursework Step 1 - Sequential Implementation
 * -------------------------------------------------------
 * Sequential Programming for Jacobi 20x20 grid problem
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For measuring execution time

// Boundary temperatures for the 2D grid
#define TOP_TEMP 15.0    // Top boundary temperature
#define BOTTOM_TEMP 60.0 // Bottom boundary temperature
#define LEFT_TEMP 47.0   // Left boundary temperature
#define RIGHT_TEMP 100.0 // Right boundary temperature

int main(int argc, char *argv[])
{
    // Check command line arguments
    if (argc != 4)
    {
        printf("Usage: %s m n tolerance\n", argv[0]);
        return 1;
    }

    // Get grid dimensions and tolerance from command line
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    double tol = atof(argv[3]);

    // Dynamically allocate 2D arrays for temperature grids
    double **t = (double **)malloc((m + 2) * sizeof(double *));
    double **tnew = (double **)malloc((m + 2) * sizeof(double *));

    // Allocate memory for each row
    for (int i = 0; i < m + 2; i++)
    {
        t[i] = (double *)malloc((n + 2) * sizeof(double));
        tnew[i] = (double *)malloc((n + 2) * sizeof(double));
    }

    printf("%d %d %lf\n", m, n, tol);

    // Initialize all grid points to 30°C
    for (int i = 0; i <= m + 1; i++)
    {
        for (int j = 0; j <= n + 1; j++)
        {
            t[i][j] = 30.0;
        }
    }

    // Set boundary conditions
    for (int i = 1; i <= m; i++)
    {
        t[i][0] = LEFT_TEMP;      // Left boundary set to 47°C
        t[i][n + 1] = RIGHT_TEMP; // Right boundary set to 100°C
    }
    for (int j = 1; j <= n; j++)
    {
        t[0][j] = TOP_TEMP;        // Top boundary set to 15°C
        t[m + 1][j] = BOTTOM_TEMP; // Bottom boundary set to 60°C
    }

    // Set corner temperatures as average of adjacent boundaries
    t[0][0] = (TOP_TEMP + LEFT_TEMP) / 2.0;             // Top-left corner
    t[0][n + 1] = (TOP_TEMP + RIGHT_TEMP) / 2.0;        // Top-right corner
    t[m + 1][0] = (BOTTOM_TEMP + LEFT_TEMP) / 2.0;      // Bottom-left corner
    t[m + 1][n + 1] = (BOTTOM_TEMP + RIGHT_TEMP) / 2.0; // Bottom-right corner

    // Start timing the execution
    clock_t start = clock();

    // Main iteration loop
    int iter = 0;
    double difmax = 1000000.0;

    while (difmax > tol)
    {
        iter++;
        difmax = 0.0;

        // Calculate new temperatures using Jacobi method
        for (int i = 1; i <= m; i++)
        {
            for (int j = 1; j <= n; j++)
            {
                tnew[i][j] = (t[i - 1][j] + t[i + 1][j] + t[i][j - 1] + t[i][j + 1]) / 4.0;
            }
        }

        // Update temperatures and find maximum difference
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

    // Calculate execution time
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print results
    printf("iter = %d  difmax = %9.11lf\n", iter, difmax);
    // Print grid values for small grids (10x10 or smaller)
    if (m <= 10 && n <= 10)
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
    printf("Execution time: %f seconds\n", cpu_time_used);

    // Free allocated memory
    for (int i = 0; i < m + 2; i++)
    {
        free(t[i]);
        free(tnew[i]);
    }
    free(t);
    free(tnew);

    return 0;
}