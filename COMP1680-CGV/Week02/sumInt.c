#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

int main(int argc, char **argv) {
    //The variables for the start and stop timer
    struct timeval startTime, stopTime;

    // This variable will hold the total time
    long totalTime;

    //loop variable
    int i;

    //variable to store the sum of integers
    int sum;

    // Start timer: get current time and store it in variable startTime
    gettimeofday(&startTime, NULL);

    //Initialize sum
    sum = 0;

    for (i=0; i<100000; i++) {
        sum = sum + i;
    }
    // Stop timer: get current time and store it in variable stopTime
    gettimeofday(&stopTime, NULL);

    // Calculate total time by subracting the startTime from the stopTime (result is in microseconds)
    totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) - (startTime.tv_sec * 1000000 + startTime.tv_usec);

    // Print the totalTime as a long integer (%ld)
    printf("The total time is %ld\n", totalTime);
    printf("The sum is %d\n",sum);
}