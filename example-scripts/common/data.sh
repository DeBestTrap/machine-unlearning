#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <dataset_path> <shards> <ul_reqs> <container_name> <report_name> [hyperparams_json]"
    exit 1
fi

dataset_path=$1
shards=$2
ul_reqs=$3
container_name=$4
report_name=$5
hyperparams_json="${6:--}"
hyperparams_csv="${hyperparams_json//\"/\"\"}"

if [[ ! -f "${report_name}.csv" ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time,hyperparams_json" > "${report_name}.csv"
fi

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

echo "data.sh"

for j in $(seq 0 $((ul_reqs))); do
    r=$((${j}))
    acc=$(python aggregation.py --strategy uniform --container "${container_name}" --shards "${shards}" --dataset "${dataset_path}" --label "${r}")
    cat containers/"${container_name}"/times/shard-*:"${r}".time > "containers/${container_name}/times/times"
    time=$(python time.py --container "${container_name}" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time},\"${hyperparams_csv}\"" >> "${report_name}.csv"
done
