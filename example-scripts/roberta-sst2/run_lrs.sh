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
slices=1
epochs=15
batch_size=32
lrs=(1e-3 1e-4 1e-5)
optimizer="adamw"
chkpt_interval=1
predict_batch_size=32
report_name="report-${model}"
dropout_rate=0.1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for lr in "${lrs[@]}"; do
    container_name="${shards}_${lr}"
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
        "${report_name}" \
        "${dropout_rate}"
done
