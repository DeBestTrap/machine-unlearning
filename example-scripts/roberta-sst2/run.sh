#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <shards> <ul_reqs>"
    exit 1
fi

shards=$1
ul_reqs=$2

datasetfile="datasets/sst2/datasetfile"
model="roberta"
container_name="${shards}"
slices=1
epochs=1
batch_size=32
learning_rate=1e-4
optimizer="adamw"
chkpt_interval=1
predict_batch_size=32
dropout_rate=0.1
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
    "${report_name}" \
    "${dropout_rate}"
