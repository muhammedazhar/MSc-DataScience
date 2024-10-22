#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int n;
    int i;
    int a[10000];
    long sum = 0;

    // We use atoi() to read an integer command line argument
    n = atoi(argv[1]);

    for (i = 0; i < n; i++) {
        a[i] = (i + 1) * (i + 1);
        printf("%d, %d\n", i + 1, a[i]);
        sum += a[i];
    }

    printf("The sum of %d squares is %ld\n", n, sum);

    return 0;
}
