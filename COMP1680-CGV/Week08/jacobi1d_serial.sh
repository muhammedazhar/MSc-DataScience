#!/bin/bash
#
#SBATCH --job-name=Jacobi1D_serial
#SBATCH --output=./Jacobi1D_serial.txt
#
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

./jacobi1d_serial.out 100 0.00001