#!/bin/bash

# Compile the code with optimization
echo -e "Compiling 'jacobi2d-Step2.c' OpenMP C program with C99 standard...\n"
gcc -std=c99 -fopenmp jacobi2d-Step2.c -o jacobi2d-Step2

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting performance tests..."
echo "-----------------------------"

# Loop over different matrix sizes
for size in 150 200 250; do
    # Loop over different thread counts
    for threads in 1 2 4 8; do
        # Print test configuration
        echo "Testing grid size of ${size}x${size} with ${threads} threads"
        # Run the program with specified size and threads
        OMP_NUM_THREADS=$threads ./jacobi2d-Step2 $size $size 0.000100
        # Separator for readability
        echo "--------------------------------------"
    done
done

echo "Testing complete!"