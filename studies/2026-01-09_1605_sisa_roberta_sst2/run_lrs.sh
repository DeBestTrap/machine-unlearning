#!/usr/bin/env bash
set -euo pipefail
study_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
if [[ $# -ge 2 && "$1" != --* ]]; then
    shards=$1
    requests=$2
    shift 2
    set -- --shards "$shards" --requests "$requests" "$@"
fi
python "$study_dir/run.py" --mode lrs "$@"
