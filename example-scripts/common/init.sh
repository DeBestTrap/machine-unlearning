#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <dataset_path> <shards> <ul_reqs> <container_name>"
    exit 1
fi

dataset_path=$1
shards=$2
ul_reqs=$3
container_name=$4

if [[ ! -d "containers/${container_name}" ]] ; then
    mkdir -p "containers/${container_name}"
    mkdir "containers/${container_name}/cache"
    mkdir "containers/${container_name}/times"
    mkdir "containers/${container_name}/outputs"
    echo 0 > "containers/${container_name}/times/null.time"
fi

if ((ul_reqs < 0)); then
    echo "Error can't be below 1"
    exit 1
fi

echo "init.sh"

if [[ -n "${DELETION_REQUEST_MANIFEST:-}" ]]; then
    # In shared mode ul_reqs counts batches/steps, including for batched deletions.
    python -c 'import json, sys; n = len(json.load(open(sys.argv[1]))["batches"]); sys.exit(0 if n == int(sys.argv[2]) else "ul_reqs must equal the shared manifest batch count")' "${DELETION_REQUEST_MANIFEST}" "${ul_reqs}"
    python distribution.py --shards "${shards}" --distribution uniform --container "${container_name}" --dataset "${dataset_path}" --label 0 --request-manifest "${DELETION_REQUEST_MANIFEST}"
    exit 0
fi

python distribution.py --shards "${shards}" --distribution uniform --container "${container_name}" --dataset "${dataset_path}" --label 0

if ((ul_reqs >= 1)); then
    for j in $(seq 1 $((ul_reqs))); do
        r=$((${j}))
        echo "${r}"
        python distribution.py --requests "${r}" --distribution uniform --container "${container_name}" --dataset "${dataset_path}" --label "${r}"
    done
fi
