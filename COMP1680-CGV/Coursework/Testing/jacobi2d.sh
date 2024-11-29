#!/bin/bash
#SBATCH --job-name=Coursework
#SBATCH --output=./Coursework.txt
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-omp

# Setting OpenMP threading environment variables
export OMP_NUM_THREADS=8

# Run the Jacobi 2D program with 0.0001 tolerance
./jacobi2d-Step3 100 100 0.0001
