#!/usr/bin/env bash
set -euo pipefail
study_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
if [[ $# -ge 1 && "$1" != --* ]]; then
    requests=$1
    shift
    set -- --requests "$requests" "$@"
fi
python "$study_dir/run.py" --mode shards "$@"
