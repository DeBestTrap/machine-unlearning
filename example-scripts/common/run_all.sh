#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

if [[ $# -lt 13 ]]; then
    echo "Usage: $0 <model> <dataset_path> <shards> <ul_reqs> <container_name> <slices> <epochs> <batch_size> <learning_rate> <optimizer> <chkpt_interval> <predict_batch_size> <report_name> [dropout_rate] [--reverse-predict-order]"
    exit 1
fi

reverse_predict=0
last_arg="${!#}"
if [[ "${last_arg}" == "--reverse-predict-order" ]]; then
    reverse_predict=1
    set -- "${@:1:$(($#-1))}"
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

predict_args=()
if [[ ${reverse_predict} -eq 1 ]]; then
    predict_args+=(--reverse-order)
fi

python "${script_dir}/predict.py" \
    "${model}" \
    "${dataset_path}" \
    "${shards}" \
    "${ul_reqs}" \
    "${container_name}" \
    "${predict_batch_size}" \
    "${predict_args[@]}"

python "${script_dir}/data.py" \
    "${dataset_path}" \
    "${shards}" \
    "${ul_reqs}" \
    "${container_name}" \
    "${report_name}" \
    "${hparams}"
