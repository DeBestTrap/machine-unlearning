#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 <dataset_path> <shards> <ul_reqs> <container_name> <report_name> [hyperparams_json]"
    exit 1
fi

dataset_path=$1
shards=$2
ul_reqs=$3
container_name=$4
report_name=$5
hyperparams_json="${6:--}"

python sisa_timing.py --container "${container_name}" --dataset "${dataset_path}" \
    --shards "${shards}" --requests "${ul_reqs}" --output "${report_name}.csv" \
    --hyperparams="${hyperparams_json}"
