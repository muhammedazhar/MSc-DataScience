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

# Determine the compiler to use
if command -v gcc-14 &> /dev/null; then
    COMPILER=gcc-14
else
    COMPILER=gcc
fi

for optmlvl in 0 1 2 3; do
    # Compile the code with different optimizations with OpenMP
    $COMPILER -std=c99 -fopenmp -O${optmlvl} jacobi2d-Step3.c -o jacobi2d-Step3-O${optmlvl}
    if [ $? -ne 0 ]; then
        echo "Compilation failed at optimization level -O${optmlvl}!"
        exit 1
    fi
done

echo "--------------------------------------"
# Loop over different matrix sizes
for size in 150 200 250; do
    # Loop over different optimization counts
    for optmlvl in 0 1 2 3; do
        # Loop over different thread levels
        for threads in 1 2 4 8; do
            # Print test configuration
            echo "Testing grid size of ${size}x${size} with ${threads} threads and optimization level -O${optmlvl}"
            # Run the program with specified size and threads
            OMP_NUM_THREADS=$threads ./jacobi2d-Step3-O${optmlvl} $size $size 0.000100
            # Separator for readability
            echo "--------------------------------------"
        done
    done
done
```

Output:

```txt
--------------------------------------
Testing grid size of 150x150 with 1 threads and optimization level -O0
iter = 20763  difmax = 0.00009999041
Execution time: 6.833578 seconds
--------------------------------------
Testing grid size of 150x150 with 2 threads and optimization level -O0
iter = 20036  difmax = 0.00009999421
Execution time: 3.642316 seconds
--------------------------------------
Testing grid size of 150x150 with 4 threads and optimization level -O0
iter = 19412  difmax = 0.00009990420
Execution time: 1.974115 seconds
--------------------------------------
Testing grid size of 150x150 with 8 threads and optimization level -O0
iter = 19558  difmax = 0.00009998308
Execution time: 1.172791 seconds
--------------------------------------
Testing grid size of 150x150 with 1 threads and optimization level -O1
iter = 20763  difmax = 0.00009999041
Execution time: 1.760912 seconds
--------------------------------------
Testing grid size of 150x150 with 2 threads and optimization level -O1
iter = 20034  difmax = 0.00009997765
Execution time: 1.153634 seconds
--------------------------------------
Testing grid size of 150x150 with 4 threads and optimization level -O1
iter = 19423  difmax = 0.00009998173
Execution time: 0.634913 seconds
--------------------------------------
Testing grid size of 150x150 with 8 threads and optimization level -O1
iter = 19404  difmax = 0.00009999061
Execution time: 0.381433 seconds
--------------------------------------
Testing grid size of 150x150 with 1 threads and optimization level -O2
iter = 20763  difmax = 0.00009999041
Execution time: 0.925734 seconds
--------------------------------------
Testing grid size of 150x150 with 2 threads and optimization level -O2
iter = 20032  difmax = 0.00009997649
Execution time: 0.668254 seconds
--------------------------------------
Testing grid size of 150x150 with 4 threads and optimization level -O2
iter = 19423  difmax = 0.00009998671
Execution time: 0.356787 seconds
--------------------------------------
Testing grid size of 150x150 with 8 threads and optimization level -O2
iter = 19387  difmax = 0.00009996107
Execution time: 0.226143 seconds
--------------------------------------
Testing grid size of 150x150 with 1 threads and optimization level -O3
iter = 20763  difmax = 0.00009999041
Execution time: 0.869270 seconds
--------------------------------------
Testing grid size of 150x150 with 2 threads and optimization level -O3
iter = 20033  difmax = 0.00009999722
Execution time: 0.632677 seconds
--------------------------------------
Testing grid size of 150x150 with 4 threads and optimization level -O3
iter = 19424  difmax = 0.00009998556
Execution time: 0.339281 seconds
--------------------------------------
Testing grid size of 150x150 with 8 threads and optimization level -O3
iter = 19730  difmax = 0.00009999910
Execution time: 0.220620 seconds
--------------------------------------
Testing grid size of 200x200 with 1 threads and optimization level -O0
iter = 32108  difmax = 0.00009999248
Execution time: 18.040790 seconds
--------------------------------------
Testing grid size of 200x200 with 2 threads and optimization level -O0
iter = 30807  difmax = 0.00009999068
Execution time: 9.307959 seconds
--------------------------------------
Testing grid size of 200x200 with 4 threads and optimization level -O0
iter = 29749  difmax = 0.00009998986
Execution time: 4.914621 seconds
--------------------------------------
Testing grid size of 200x200 with 8 threads and optimization level -O0
iter = 30392  difmax = 0.00009998702
Execution time: 2.916617 seconds
--------------------------------------
Testing grid size of 200x200 with 1 threads and optimization level -O1
iter = 32108  difmax = 0.00009999248
Execution time: 4.718834 seconds
--------------------------------------
Testing grid size of 200x200 with 2 threads and optimization level -O1
iter = 30799  difmax = 0.00009999040
Execution time: 2.570721 seconds
--------------------------------------
Testing grid size of 200x200 with 4 threads and optimization level -O1
iter = 29717  difmax = 0.00009999629
Execution time: 1.528857 seconds
--------------------------------------
Testing grid size of 200x200 with 8 threads and optimization level -O1
iter = 29989  difmax = 0.00009998269
Execution time: 0.902573 seconds
--------------------------------------
Testing grid size of 200x200 with 1 threads and optimization level -O2
iter = 32108  difmax = 0.00009999248
Execution time: 2.560066 seconds
--------------------------------------
Testing grid size of 200x200 with 2 threads and optimization level -O2
iter = 30802  difmax = 0.00009998338
Execution time: 1.450910 seconds
--------------------------------------
Testing grid size of 200x200 with 4 threads and optimization level -O2
iter = 29705  difmax = 0.00009998981
Execution time: 0.932664 seconds
--------------------------------------
Testing grid size of 200x200 with 8 threads and optimization level -O2
iter = 31783  difmax = 0.00009999760
Execution time: 0.539187 seconds
--------------------------------------
Testing grid size of 200x200 with 1 threads and optimization level -O3
iter = 32108  difmax = 0.00009999248
Execution time: 2.423073 seconds
--------------------------------------
Testing grid size of 200x200 with 2 threads and optimization level -O3
iter = 30811  difmax = 0.00009998999
Execution time: 1.366636 seconds
--------------------------------------
Testing grid size of 200x200 with 4 threads and optimization level -O3
iter = 30818  difmax = 0.00009992472
Execution time: 0.829421 seconds
--------------------------------------
Testing grid size of 200x200 with 8 threads and optimization level -O3
iter = 29988  difmax = 0.00009998336
Execution time: 0.483511 seconds
--------------------------------------
Testing grid size of 250x250 with 1 threads and optimization level -O0
iter = 44398  difmax = 0.00009999424
Execution time: 38.671279 seconds
--------------------------------------
Testing grid size of 250x250 with 2 threads and optimization level -O0
iter = 42381  difmax = 0.00009999290
Execution time: 19.025638 seconds
--------------------------------------
Testing grid size of 250x250 with 4 threads and optimization level -O0
iter = 42321  difmax = 0.00009988293
Execution time: 10.780990 seconds
--------------------------------------
Testing grid size of 250x250 with 8 threads and optimization level -O0
iter = 42684  difmax = 0.00009999864
Execution time: 6.312247 seconds
--------------------------------------
Testing grid size of 250x250 with 1 threads and optimization level -O1
iter = 44398  difmax = 0.00009999424
Execution time: 10.182798 seconds
--------------------------------------
Testing grid size of 250x250 with 2 threads and optimization level -O1
iter = 42372  difmax = 0.00009999710
Execution time: 5.149216 seconds
--------------------------------------
Testing grid size of 250x250 with 4 threads and optimization level -O1
iter = 42295  difmax = 0.00009999975
Execution time: 3.203073 seconds
--------------------------------------
Testing grid size of 250x250 with 8 threads and optimization level -O1
iter = 44204  difmax = 0.00009999342
Execution time: 1.933559 seconds
--------------------------------------
Testing grid size of 250x250 with 1 threads and optimization level -O2
iter = 44398  difmax = 0.00009999424
Execution time: 5.311001 seconds
--------------------------------------
Testing grid size of 250x250 with 2 threads and optimization level -O2
iter = 42383  difmax = 0.00009998860
Execution time: 2.955308 seconds
--------------------------------------
Testing grid size of 250x250 with 4 threads and optimization level -O2
iter = 42328  difmax = 0.00009994597
Execution time: 1.809079 seconds
--------------------------------------
Testing grid size of 250x250 with 8 threads and optimization level -O2
iter = 41784  difmax = 0.00009999386
Execution time: 0.981445 seconds
--------------------------------------
Testing grid size of 250x250 with 1 threads and optimization level -O3
iter = 44398  difmax = 0.00009999424
Execution time: 5.051395 seconds
--------------------------------------
Testing grid size of 250x250 with 2 threads and optimization level -O3
iter = 42314  difmax = 0.00009999638
Execution time: 2.717689 seconds
--------------------------------------
Testing grid size of 250x250 with 4 threads and optimization level -O3
iter = 40698  difmax = 0.00009999491
Execution time: 1.721731 seconds
--------------------------------------
Testing grid size of 250x250 with 8 threads and optimization level -O3
iter = 40087  difmax = 0.00009999116
Execution time: 0.882655 seconds
--------------------------------------
```

This is the final step. All the chat before was step 1 & 2. Based on the all that context as well as the givn Step 3 context, can you write a report for this step?
