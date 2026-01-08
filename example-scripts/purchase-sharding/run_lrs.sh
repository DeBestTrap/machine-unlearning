#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
ul_reqs=$2

datasetfile="datasets/purchase/datasetfile"
model="purchase"
batch_size=32
slices=1
lrs=(1e-6 3e-6 1e-5)
epochs=1
optimizer="sgd"
chkpt_interval=1
predict_batch_size=64
report_name="report-${model}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for lr in "${lrs[@]}"; do
    container_name="${shards}_${lr}"
    echo "${container_name}"
    bash "${script_dir}/../common/run_all.sh" \
        "${model}" \
        "${datasetfile}" \
        "${shards}" \
        "${ul_reqs}" \
        "${container_name}" \
        "${slices}" \
        "${epochs}" \
        "${batch_size}" \
        "${lr}" \
        "${optimizer}" \
        "${chkpt_interval}" \
        "${predict_batch_size}" \
        "${report_name}"
done
