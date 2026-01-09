#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
ul_reqs=$2

datasetfile="datasets/purchase/datasetfile"
model="purchase"
container_name="p-${shards}"
batch_size=32
slices=32
epochs=15
learning_rate=0.001
optimizer="sgd"
chkpt_interval=1
predict_batch_size=32
report_name="report-${model}"

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
    "${report_name}"
