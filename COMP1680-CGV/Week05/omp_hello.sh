#!/bin/bash
#
#SBATCH --job-name=OMP_Hello
#SBATCH --output=./omp_hello.txt
#
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

./omp_hello.out 2