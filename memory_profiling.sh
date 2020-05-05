#!/bin/bash
#require installation of valgrind
#sudo apt update
#sudo apt install valgrind

NUM_CORE=4

exe_name="xtensor_lasso"

run_command="mpirun -n $NUM_CORE ./$exe_name"

rm memory_usage.txt

touch memory_usage.txt

echo "Peak memory usage (in MB) is:" >> memory_usage.txt

for num_data in 400 2000 4000
do
	usages_list=""
	for num_feature in 100 200 400
	do 	
		rm -rf ./data/*
		# 20 % non_zero feature
		python3.6 generate_lasso_data.py $num_data $num_feature $(expr $num_feature / 5) $NUM_CORE
		
		rm -rf massif.*
		#profile the memory use the executable and output the result to massif.out
		valgrind --tool=massif --pages-as-heap=yes --massif-out-file=massif.out $run_command

		echo "Peak memory usage (in MB) is:"
		#print the peak memory usage (B)
		memory_usage=$(grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1)

		memory_usage=$(expr $memory_usage / 1000000)

		echo $memory_usage

		usages_list="$usages_list $memory_usage"
	done
	
	echo $usages_list >> memory_usage.txt

done
