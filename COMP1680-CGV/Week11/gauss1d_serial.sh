#!/bin/bash
#
#SBATCH --job-name=Gauss1D_serial
#SBATCH --output=./Gauss1D_serial.txt
#
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

./gauss1d_serial.out 100 0.00001