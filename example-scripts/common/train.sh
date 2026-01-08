#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 11 ]]; then
    echo "Usage: $0 <model> <dataset_path> <shards> <ul_reqs> <container_name> <slices> <epochs> <batch_size> <learning_rate> <optimizer> <chkpt_interval> [dropout_rate]"
    exit 1
fi

model=$1
dataset_path=$2
shards=$3
ul_reqs=$4
container_name=$5
slices=$6
epochs=$7
batch_size=$8
learning_rate=$9
optimizer=${10}
chkpt_interval=${11}
dropout_rate=${12:-}

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

echo "train.sh"

for i in $(seq 0 "$((${shards}-1))"); do
    for j in $(seq 0 $((ul_reqs))); do
        echo "shard: $((${i}+1))/${shards}, requests: $((${j}+1))/$((${ul_reqs}+1))"
        r=$((${j}))
        dropout_args=()
        if [[ -n "${dropout_rate}" ]]; then
            dropout_args=(--dropout_rate "${dropout_rate}")
        fi
        python sisa.py --model "${model}" --train --slices "${slices}" --dataset "${dataset_path}" --label "${r}" --epochs "${epochs}" --batch_size "${batch_size}" --learning_rate "${learning_rate}" --optimizer "${optimizer}" --chkpt_interval "${chkpt_interval}" --container "${container_name}" --shard "${i}" "${dropout_args[@]}"
    done
done
