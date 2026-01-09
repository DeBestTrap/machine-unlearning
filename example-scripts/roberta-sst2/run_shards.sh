#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ul_reqs>"
    exit 1
fi

ul_reqs=$1
num_shards=(1 4 16)

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for shards in "${num_shards[@]}"; do
    bash "${script_dir}/run.sh" "${shards}" "${ul_reqs}"
done