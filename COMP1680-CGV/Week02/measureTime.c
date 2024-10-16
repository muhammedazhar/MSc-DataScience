#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

int main(int argc, char** argv) {
    //The variables for the start and stop timer
    struct timeval startTime, stopTime;

    //loop variable
    int i;

    // This variable will hold the total time
    long totalTime;

    // Start timer: get current time and store it in variable startTime
    gettimeofday(&startTime, NULL);

    for (i = 0; i < 10000; i++) {
    printf("hello world\n");
    }

    // Stop timer: get current time and store it in variable stopTime
    gettimeofday(&stopTime, NULL);

    // Calculate total time by subracting the startTime from the stopTime (result is in microseconds)
    totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) - (startTime.tv_sec * 1000000 + startTime.tv_usec);

    // Print the totalTime as a long integer (%ld)
    printf("%ld\n", totalTime);

    return (0);
}