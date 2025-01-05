#!/usr/bin/bash
#SBATCH --job-name=mpi
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=COMP1680-mpi
#SBATCH --output=MPI_Tutorial.txt

module load OpenMPI/4.1.5-GCC-12.3.0

echo "Hello World"
mpicc Hello_World2.c
mpirun -n 8 ./a.out
echo ""

echo "Ping Pong"
mpicc Ping_pong10.c
mpirun -n 2 ./a.out
echo ""

echo "Forward Backward"
mpicc Forward_Backward12.c
mpirun 0 -n 4 ./a.out
echo ""

echo "Odd Even"
mpicc Odd_Even13.c
mpirun -n 4 ./a.out
echo ""

echo "All Reduce"
mpicc All_Reduce16.c
mpirun -n 4 ./a.out
echo ""