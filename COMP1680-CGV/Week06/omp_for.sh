#!/bin/bash
#
#SBATCH --job-name=OMP_FOR
#SBATCH --output=./omp_for.txt
#
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

./omp_for.out 4 16
./omp_for_dynamic.out 4 16