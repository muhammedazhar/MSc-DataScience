#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv) {
    // The variables for the start and stop timer
    struct timeval startTime, stopTime;

    // This variable will hold the total time
    long totalTime;

    // loop variable
    int i, j;

    // variable to store the sum of integers
    long sum;

    // Start timer: get current time and store it in variable startTime
    gettimeofday(&startTime, NULL);

    for (j = 100000; j <= 1000000; j += 100000) {
        // Initialize sum
        sum = 0;

        for (i = 0; i < j; i++) {
            sum = sum + i;
        }
        // Stop timer: get current time and store it in variable stopTime
        gettimeofday(&stopTime, NULL);

        // Calculate total time by subracting the startTime from the stopTime
        // (result is in microseconds)
        totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) -
                    (startTime.tv_sec * 1000000 + startTime.tv_usec);

        // Print the totalTime as a long integer (%ld)
        printf("Number of iterations in %d\n", j);
        printf("The total time is %ld\n", totalTime);
        printf("The sum is %ld\n", sum);
    }
}
