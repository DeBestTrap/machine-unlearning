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
slices=4
epochs=15
batch_size=32
learning_rate=0.00001
optimizer="adam"
chkpt_interval=1
predict_batch_size=16

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
bash "${script_dir}/../common/init.sh" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}"
bash "${script_dir}/../common/train.sh" "${model}" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}" "${slices}" "${epochs}" "${batch_size}" "${learning_rate}" "${optimizer}" "${chkpt_interval}"
bash "${script_dir}/../common/predict.sh" "${model}" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}" "${predict_batch_size}"
bash "${script_dir}/../common/data.sh" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}"
