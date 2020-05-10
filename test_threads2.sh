#!/bin/bash

for j in {6..2}
do
Nprocs=$((2**$j))
echo Using $Nprocs MPI processes

for i in {4..0}
do
    N=$((2**$i))
    echo Using $N threads

    export OMP_NUM_THREADS=$N
    
    mpirun -np $Nprocs ./build/lasso_reg_path >> "$SCRATCH"/cs205/group17/perf_results/output_"$Nprocs"_"$N".txt

done
done
