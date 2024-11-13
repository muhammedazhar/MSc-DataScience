#!/bin/bash
#
#SBATCH --job-name=OMP_AddArray
#SBATCH --output=./omp_addarray.txt
#
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

./omp_addarray.out 4 100 10