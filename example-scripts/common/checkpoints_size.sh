#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <container_name>"
    exit 1
fi

# Persist this measurement locally so it survives checkpoint cleanup.
python container_storage.py size "$1"
