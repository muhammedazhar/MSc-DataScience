#!/bin/bash
#SBATCH --job-name=jacobi2d-Step3      # Job name
#SBATCH --output=./jacobi2d-Step3.txt  # Output file
#SBATCH --cpus-per-task=8              # Number of CPUs per task
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --partition=COMP1680-omp       # Partition name

# Determine the compiler to use
if command -v gcc-14 &> /dev/null; then
    COMPILER=gcc-14
else
    COMPILER=gcc
fi

for optmlvl in 0 1 2 3; do
    # Compile the code with different optimizations with OpenMP
    $COMPILER -std=c99 -fopenmp -O${optmlvl} jacobi2d-Step3.c -o jacobi2d-Step3-O${optmlvl}
    if [ $? -ne 0 ]; then
        echo "Compilation failed at optimization level -O${optmlvl}!"
        exit 1
    fi
done

echo "--------------------------------------"
# Loop over different matrix sizes
for size in 150 200 250; do
    # Loop over different optimization levels
    for optmlvl in 0 1 2 3; do
        # Loop over different thread counts
        for threads in 1 2 4 8; do
            # Print test configuration
            echo "Testing grid size of ${size}x${size} with ${threads} threads and optimization level -O${optmlvl}"
            # Run the program with specified size and threads
            OMP_NUM_THREADS=$threads ./jacobi2d-Step3-O${optmlvl} $size $size 0.000100
            # Separator for readability
            echo "--------------------------------------"
        done
    done
done
