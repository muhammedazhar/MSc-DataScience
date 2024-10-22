#!/bin/bash
#
#SBATCH --job-name=Arguments
#SBATCH --output=./arguments.txt
#
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

gcc arguments.c -o arguments.out
./arguments.out 2 3 3.141
