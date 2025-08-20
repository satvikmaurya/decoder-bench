#!/bin/bash

dataset_dir="../datasets"
num_shots=1000000
batch_size=1000
output_file="./results/benchmark_results.csv"

mkdir -p ./results

find_p_values() {
    local pattern=$1
    find "$dataset_dir" -name "$pattern" | grep -oE "_0\.[0-9]+" | sort -u | sed 's/_//'
}

run_benchmark() {
    local filename=$1
    local decoder=$2
    local decoder_params=$3
    
    echo "=================================================="
    echo "Running benchmark for: $filename"
    echo "Decoder: $decoder"
    echo "=================================================="
    
    cmd="python eval.py --filename $filename --dataset_dir $dataset_dir --decoder $decoder --num_shots $num_shots --num_workers 32 --batch_size $batch_size --output_file $output_file"
    
    if [ ! -z "$decoder_params" ]; then
        cmd="$cmd --decoder_params '$decoder_params'"
    fi
    
    eval $cmd
    
    echo ""
    echo ""
}

color_p_values=$(find_p_values "color_circuit_7_*.h5")

surface7_p_values=$(find_p_values "surface_circuit_7_*_z.h5")
surface11_p_values=$(find_p_values "surface_circuit_11_*_z.h5")

qldpc6_p_values=$(find_p_values "qldpc_circuit_6_*_z.h5")
qldpc12_p_values=$(find_p_values "qldpc_circuit_12_*_z.h5")

ls11_p_values=$(find_p_values "ls_circuit_11_*_z.h5")

echo "Starting benchmarks..."

for p in $color_p_values; do
    filename="color_circuit_7_${p}.h5"
    if [ -f "$dataset_dir/$filename" ]; then
        run_benchmark "$filename" "belieffind" ""
        run_benchmark "$filename" "bplsd" ""
    fi
done

for p in $surface7_p_values; do
    filename="surface_circuit_7_${p}_z.h5"
    if [ -f "$dataset_dir/$filename" ]; then
        run_benchmark "$filename" "pymatching" ""
        run_benchmark "$filename" "belieffind" ""
    fi
done

for p in $surface11_p_values; do
    filename="surface_circuit_11_${p}_z.h5"
    if [ -f "$dataset_dir/$filename" ]; then
        run_benchmark "$filename" "pymatching" ""
        run_benchmark "$filename" "belieffind" ""
    fi
done

for p in $qldpc6_p_values; do
    filename="qldpc_circuit_6_${p}_z.h5"
    if [ -f "$dataset_dir/$filename" ]; then
        run_benchmark "$filename" "belieffind" ""
        run_benchmark "$filename" "bplsd" ""
    fi
done

for p in $qldpc12_p_values; do
    filename="qldpc_circuit_12_${p}_z.h5"
    if [ -f "$dataset_dir/$filename" ]; then
        run_benchmark "$filename" "belieffind" ""
        run_benchmark "$filename" "bplsd" ""
    fi
done

for p in $ls11_p_values; do
    filename="ls_circuit_11_${p}_z.h5"
    if [ -f "$dataset_dir/$filename" ]; then
        run_benchmark "$filename" "pymatching" ""
        run_benchmark "$filename" "belieffind" ""
    fi
done

echo "Benchmarking complete! Results saved to $output_file"
