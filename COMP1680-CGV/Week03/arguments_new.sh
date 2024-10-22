#!/bin/bash
#
#SBATCH --job-name=Arguments
#SBATCH --output=./arguments.txt
#
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=COMP1680-dev

gcc arguments_new.c -o arugments_new.out
./arguments_new.out 100
./arguments_new.out 200
./arguments_new.out 300
./arguments_new.out 400
./arguments_new.out 500
./arguments_new.out 600
