#!/bin/bash

# Determine the compiler to use
if command -v gcc-14 &> /dev/null; then
    COMPILER=gcc-14
else
    COMPILER=gcc
fi

# Compile the code with different thread counts
echo "Compiling 'jacobi2d-Step2.c' with C99 standard and different threads counts using $COMPILER..."
$COMPILER -std=c99 -fopenmp jacobi2d-Step2.c -o jacobi2d-Step2

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting performance tests..."
echo "--------------------------------------" >> jacobi2d-Step2.txt

# Loop over different matrix sizes
for size in 20; do
    # Loop over different thread counts
    for threads in 1 2 4 8; do
        # Print test configuration
        echo "Testing grid size of ${size}x${size} with ${threads} threads" >> jacobi2d-Step2.txt
        # Run the program with specified size and threads
        OMP_NUM_THREADS=$threads ./jacobi2d-Step2 $size $size 0.000100 >> jacobi2d-Step2.txt
        # Separator for readability
        echo "--------------------------------------" >> jacobi2d-Step2.txt
    done
done

echo "Testing complete!"