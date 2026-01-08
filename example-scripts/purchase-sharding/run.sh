#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

shards=$1
ul_reqs=$2

datasetfile="datasets/purchase/datasetfile"
model="purchase"
batch_size=32
slices=32
epochs=1
batch_size=16
learning_rate=0.001
optimizer="sgd"
chkpt_interval=1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
container_name="${shards}-3"
bash "${script_dir}/../common/init.sh" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}"
bash "${script_dir}/../common/train.sh" "${model}" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}" "${slices}" "${epochs}" "${batch_size}" "${learning_rate}" "${optimizer}" "${chkpt_interval}"
bash "${script_dir}/../common/predict.sh" "${model}" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}" "${batch_size}"
bash "${script_dir}/../common/data.sh" "${datasetfile}" "${shards}" "${ul_reqs}" "${container_name}"