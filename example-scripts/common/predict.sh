#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <model> <dataset_path> <shards> <ul_reqs> <container_name> <batch_size>"
    exit 1
fi

model=$1
dataset_path=$2
shards=$3
ul_reqs=$4
container_name=$5
batch_size=$6

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

echo "predict.sh"

for i in $(seq 0 "$((${shards}-1))"); do
    for j in $(seq 0 $((ul_reqs))); do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/$((${ul_reqs}+1))"
        r=$((${j}))
        python sisa.py --model "${model}" --test --dataset "${dataset_path}" --label "${r}" --batch_size "${batch_size}" --container "${container_name}" --shard "${i}"
    done
done
