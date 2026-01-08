#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <dataset_path> <shards> <ul_reqs> <container_name>"
    exit 1
fi

dataset_path=$1
shards=$2
ul_reqs=$3
container_name=$4

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > general-report.csv
fi

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

echo "data.sh"

for j in $(seq 0 $((ul_reqs))); do
    r=$((${j}))
    acc=$(python aggregation.py --strategy uniform --container "${container_name}" --shards "${shards}" --dataset "${dataset_path}" --label "${r}")
    cat containers/"${shards}"/times/shard-*:"${r}".time > "containers/${container_name}/times/times"
    time=$(python time.py --container "${container_name}" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time}" >> general-report.csv
done
