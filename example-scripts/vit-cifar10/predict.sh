#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
ul_reqs=$2

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

for i in $(seq 0 "$((${shards}-1))"); do
    for j in $(seq 0 $((ul_reqs))); do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/$((${ul_reqs}+1))"
        r=$((${j}))
        python sisa.py --model vit --test --dataset datasets/cifar10/datasetfile --label "${r}" --batch_size 16 --container "${shards}" --shard "${i}"
    done
done
