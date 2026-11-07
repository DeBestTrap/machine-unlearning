#!/usr/bin/env bash
set -euo pipefail
study_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
exec python "$study_dir/run.py" "$@"
