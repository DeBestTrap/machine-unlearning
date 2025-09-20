#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
ul_reqs=$2
    
if [[ ! -d "containers/${shards}" ]] ; then
    mkdir "containers/${shards}"
    mkdir "containers/${shards}/cache"
    mkdir "containers/${shards}/times"
    mkdir "containers/${shards}/outputs"
    echo 0 > "containers/${shards}/times/null.time"
fi

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

python distribution.py --shards "${shards}" --distribution uniform --container "${shards}" --dataset datasets/cifar10/datasetfile --label 0

if ((ul_reqs >= 1)); then
    for j in $(seq 1 $((ul_reqs))); do
        r=$((${j}))
        echo "${r}"
        python distribution.py --requests "${r}" --distribution uniform --container "${shards}" --dataset datasets/cifar10/datasetfile --label "${r}"
    done
fi
