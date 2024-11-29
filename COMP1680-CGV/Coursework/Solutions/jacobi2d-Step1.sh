#!/bin/bash

# Compile the code with optimization
echo -e "Compiling 'jacobi2d-Step1.c' program with C99 standard and different optimizations...\n"
# Explicitly no optimization to avoid pontential overrides
gcc -std=c99 -O0 jacobi2d-Step1.c -o jacobi2d-Step1-O0
gcc -std=c99 -O1 jacobi2d-Step1.c -o jacobi2d-Step1-O1
gcc -std=c99 -O2 jacobi2d-Step1.c -o jacobi2d-Step1-O2
gcc -std=c99 -O3 jacobi2d-Step1.c -o jacobi2d-Step1-O3

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting performance tests..."
echo "-----------------------------"

# Loop over different matrix sizes
for size in 150 200 250; do
    # Loop over different thread counts
    for optmlvl in 0 1 2 3; do
        # Print test configuration
        echo "Testing grid size of ${size}x${size} with ${optmlvl} optimization level"
        # Run the program with specified size and optimization level, redirecting output to a file
        ./jacobi2d-Step1-$optmlvl $size $size 0.000100 >> jacobi2d-Step1.txt
        # Separator for readability
        echo "--------------------------------------" >> jacobi2d-Step1.txt
    done
done

echo "Testing complete!"