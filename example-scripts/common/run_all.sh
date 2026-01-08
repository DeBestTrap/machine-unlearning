#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 13 ]]; then
    echo "Usage: $0 <model> <dataset_path> <shards> <ul_reqs> <container_name> <slices> <epochs> <batch_size> <learning_rate> <optimizer> <chkpt_interval> <predict_batch_size> <report_name> [dropout_rate]"
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
predict_batch_size=${12}
report_name=${13}
dropout_rate=${14:-}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
printf -v hparams '{"lr":%s,"batch_size":%s,"epochs":%s,"optimizer":"%s","slices":%s,"chkpt_interval":%s}' \
    "${learning_rate}" "${batch_size}" "${epochs}" "${optimizer}" "${slices}" "${chkpt_interval}"
bash "${script_dir}/init.sh" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}"
bash "${script_dir}/train.sh" "${model}" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}" "${slices}" "${epochs}" "${batch_size}" "${learning_rate}" "${optimizer}" "${chkpt_interval}" "${dropout_rate}"
bash "${script_dir}/predict.sh" "${model}" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}" "${predict_batch_size}"
bash "${script_dir}/data.sh" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}" "${report_name}" "${hparams}"
