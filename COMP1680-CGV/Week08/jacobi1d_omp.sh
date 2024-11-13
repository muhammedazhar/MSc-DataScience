#!/bin/bash
#
#SBATCH --job-name=Jacobi1D_omp
#SBATCH --output=./Jacobi1D_omp.txt
#
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

./jacobi1d_omp.out 100 0.00001 8