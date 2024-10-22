#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int a;
    int b;
    float c;

    // We use atoi() to read an integer command line argument
    a = atoi(argv[1]);
    b = atoi(argv[2]);

    // We use atof() to read a float command line argument
    c = atof(argv[3]);

    printf("Hello world!\n");

    printf("a= %d\n", a);
    printf("b= %d\n", b);
    printf("c= %f\n", c);

    return 0;
}
