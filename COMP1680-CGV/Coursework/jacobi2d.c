#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s m n tolerance\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    double tol = atof(argv[3]);
    
    // Dynamically allocate 2D arrays
    double **t = (double **)malloc((m + 2) * sizeof(double *));
    double **tnew = (double **)malloc((m + 2) * sizeof(double *));
    
    for (int i = 0; i < m + 2; i++) {
        t[i] = (double *)malloc((n + 2) * sizeof(double));
        tnew[i] = (double *)malloc((n + 2) * sizeof(double));
    }

    printf("%d %d %lf\n", m, n, tol);

    // Initialize temperature array
    for (int i = 0; i <= m + 1; i++) {
        for (int j = 0; j <= n + 1; j++) {
            t[i][j] = 30.0;
        }
    }

    // Fix boundary conditions
    for (int i = 1; i <= m; i++) {
        t[i][0] = 40.0;
        t[i][n + 1] = 90.0;
    }
    for (int j = 1; j <= n; j++) {
        t[0][j] = 30.0;
        t[m + 1][j] = 50.0;
    }

    // Main loop
    int iter = 0;
    double difmax = 1000000.0;
    
    while (difmax > tol) {
        iter++;
        difmax = 0.0;

        // Update temperature for next iteration
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                tnew[i][j] = (t[i-1][j] + t[i+1][j] + t[i][j-1] + t[i][j+1]) / 4.0;
            }
        }

        // Calculate maximum difference
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                double diff = fabs(tnew[i][j] - t[i][j]);
                difmax = (diff > difmax) ? diff : difmax;
                t[i][j] = tnew[i][j];
            }
        }
    }

    // Print results
    printf("iter = %d  difmax = %9.11lf\n", iter, difmax);
    for (int i = 0; i <= m + 1; i++) {
        for (int j = 0; j <= n + 1; j++) {
            printf("%3.5lf ", t[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory
    for (int i = 0; i < m + 2; i++) {
        free(t[i]);
        free(tnew[i]);
    }
    free(t);
    free(tnew);

    return 0;
}
