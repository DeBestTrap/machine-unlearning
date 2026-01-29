#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <shards> <ul_reqs>"
    exit 1
fi

shards=$1
ul_reqs=$2

export CUDA_VISIBLE_DEVICES=1

datasetfile="datasets/sst2/datasetfile"
model="roberta"
slices=1
epochs=1
batch_size=32
learning_rate=1e-5
optimizer="adamw"
chkpt_interval=1
predict_batch_size=128
dropout_rate=0.1
report_name="report-${model}"
container_name="${shards}-rr"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
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
    "${dropout_rate}"
# bash "${script_dir}/../common/init.sh" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}"