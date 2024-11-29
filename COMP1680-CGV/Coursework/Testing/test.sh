#!/bin/bash

# Determine the compiler to use
if command -v gcc-14 &> /dev/null; then
    COMPILER=gcc-14
else
    COMPILER=gcc
fi

# Compile the code with different optimizations
echo "Compiling 'jacobi2d-Step1.c' with C99 standard and different optimizations using $COMPILER..."

for optmlvl in 0 1 2 3; do
    $COMPILER -std=c99 -O${optmlvl} jacobi2d-Step1.c -o jacobi2d-Step1-O${optmlvl}
    if [ $? -ne 0 ]; then
        echo "Compilation failed at optimization level -O${optmlvl}!"
        exit 1
    fi
done

echo "Starting performance tests..."
echo "--------------------------------------" > jacobi2d-Step1.txt

# Loop over different matrix sizes and optimization levels
for size in 150 200 250; do
    for optmlvl in 0 1 2 3; do
        # Print test configuration
        echo "Testing grid size of ${size}x${size} with optimization level -O${optmlvl}" >> jacobi2d-Step1.txt
        # Run the program and append output to the file
        ./jacobi2d-Step1-O${optmlvl} $size $size 0.000100 >> jacobi2d-Step1.txt
        echo "--------------------------------------" >> jacobi2d-Step1.txt
    done
done

echo "Testing complete!"