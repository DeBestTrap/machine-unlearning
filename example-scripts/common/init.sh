#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "${script_dir}/init.py" "$@"
