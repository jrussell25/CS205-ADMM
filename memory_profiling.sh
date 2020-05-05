#!/bin/bash
#require installation of valgrind
#sudo apt update
#sudo apt install valgrind

NUM_CORE=4

exe_name="xtensor_lasso"

run_command="mpirun -n $NUM_CORE ./$exe_name"

#profile the memory use the executable and output the result to massif.out
valgrind --tool=massif --pages-as-heap=yes --massif-out-file=massif.out $run_command

echo "Peak memory usage (in MB) is:"
#print the peak memory usage (B)
memory_usage=$(grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1)

echo $(expr $memory_usage / 1000000)
