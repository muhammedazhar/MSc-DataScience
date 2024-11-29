# Prompt

Context: [Using the HPC and the SLURM queue you are to run performance tests with the OpenMP implementation you created in step 2. This will require that you remove most of the print output from the code and increase the problem size to provide sufficient work to demonstrate useful speedup. You are expected to provide speedup results:

• for a range of problem sizes, you are unlikely to see much speedup for small domains, use the same problem size as step 1
• for a range of number of threads (from 2 up to 8 threads) In calculating the speedup of your parallel code you should use the optimized single processor version of your code you produced in step 1 and compare to this. You will need to apply similar compiler optimizations to your parallel code. Please list your runtimes in a suitable unit.

This section is required to provide details of your implementation of steps 1 to 3 as described above. You should include discussion of your solutions and provide a clear description of; the code changes you have implemented including code snippets, your compilation and execution processes and your test cases. For step 3 you are expected to provide tabular and graphical results.]

Bash script:

```bash
#!/bin/bash
#SBATCH --job-name=jacobi2d-Step3      # Job name
#SBATCH --output=./jacobi2d-Step3.txt  # Output file
#SBATCH --cpus-per-task=8              # Number of CPUs per task
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --ntasks-per-node=1            # Number of tasks per node
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --partition=COMP1680-omp       # Partition name

echo "--------------------------------------"
# Loop over different matrix sizes
for size in 150 200 250; do
    # Loop over different thread counts
    for threads in 1 2 4 8; do
        # Print test configuration
        echo "Testing grid size of ${size}x${size} with ${threads} threads"
        # Run the program with specified size and threads
        OMP_NUM_THREADS=$threads ./jacobi2d-Step3 $size $size 0.000100
        # Separator for readability
        echo "--------------------------------------"
    done
done
```

Output:

```txt
--------------------------------------
Testing grid size of 150x150 with 1 threads
iter = 20763  difmax = 0.00009999041
Execution time: 6.823984 seconds
--------------------------------------
Testing grid size of 150x150 with 2 threads
iter = 20028  difmax = 0.00009999062
Execution time: 3.478909 seconds
--------------------------------------
Testing grid size of 150x150 with 4 threads
iter = 19602  difmax = 0.00009987131
Execution time: 1.943516 seconds
--------------------------------------
Testing grid size of 150x150 with 8 threads
iter = 19403  difmax = 0.00009999299
Execution time: 1.268807 seconds
--------------------------------------
Testing grid size of 200x200 with 1 threads
iter = 32108  difmax = 0.00009999248
Execution time: 17.983774 seconds
--------------------------------------
Testing grid size of 200x200 with 2 threads
iter = 30811  difmax = 0.00009998081
Execution time: 8.991890 seconds
--------------------------------------
Testing grid size of 200x200 with 4 threads
iter = 29752  difmax = 0.00009999490
Execution time: 4.849675 seconds
--------------------------------------
Testing grid size of 200x200 with 8 threads
iter = 29367  difmax = 0.00009999389
Execution time: 2.818227 seconds
--------------------------------------
Testing grid size of 250x250 with 1 threads
iter = 44398  difmax = 0.00009999424
Execution time: 38.797998 seconds
--------------------------------------
Testing grid size of 250x250 with 2 threads
iter = 42381  difmax = 0.00009999884
Execution time: 18.960634 seconds
--------------------------------------
Testing grid size of 250x250 with 4 threads
iter = 42366  difmax = 0.00009995415
Execution time: 10.666419 seconds
--------------------------------------
Testing grid size of 250x250 with 8 threads
iter = 41846  difmax = 0.00009999333
Execution time: 6.186029 seconds
--------------------------------------
```

This is the final step. All the chat before was step 1 & 2. Based on the all that context as well as the give Step 3 context, can you write a report for this step?
