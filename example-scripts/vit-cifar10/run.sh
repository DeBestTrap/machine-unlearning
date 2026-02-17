#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <shards> <ul_reqs> [reverse_predict]"
    exit 1
fi

shards=$1
ul_reqs=$2
reverse_predict=${3:-}

datasetfile="datasets/cifar10/datasetfile"
model="vit"
container_name="${shards}-cc"
slices=1
epochs=1
batch_size=64
learning_rate=0.00001
optimizer="adamw"
chkpt_interval=1
predict_batch_size=128
report_name="report-${model}"

# export CUDA_VISIBLE_DEVICES=1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
reverse_flag=()
if [[ "${reverse_predict}" == "1" || "${reverse_predict}" == "reverse" ]]; then
    reverse_flag=(--reverse-predict-order)
fi
bash "${script_dir}/../common/run_all.sh" \
    "${model}" \
    "${datasetfile}" \
    "${shards}" \
    "${ul_reqs}" \
    "${container_name}" \
    "${slices}" \
    "${epochs}" \
    "${batch_size}" \
    "${learning_rate}" \
    "${optimizer}" \
    "${chkpt_interval}" \
    "${predict_batch_size}" \
    "${report_name}" \
    "${reverse_flag[@]}"
