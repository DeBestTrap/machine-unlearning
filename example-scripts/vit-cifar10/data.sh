#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
ul_reqs=$2

if [[ ! -f general-report.csv ]]; then
    echo "nb_shards,nb_requests,accuracy,retraining_time" > general-report.csv
fi


if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

for j in $(seq 0 $((ul_reqs))); do
    r=$((${j}))
    acc=$(python aggregation.py --strategy uniform --container "${shards}" --shards "${shards}" --dataset datasets/cifar10/datasetfile --label "${r}")
    cat containers/"${shards}"/times/shard-*:"${r}".time > "containers/${shards}/times/times"
    time=$(python time.py --container "${shards}" | awk -F ',' '{print $1}')
    echo "${shards},${r},${acc},${time}" >> general-report.csv
done
