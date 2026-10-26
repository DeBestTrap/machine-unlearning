#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <container_name>"
    exit 1
fi

container_name=$1
cache_dir="containers/${container_name}/cache"

if [[ ! -d "${cache_dir}" ]]; then
    echo "Checkpoint cache not found at ${cache_dir}"
    echo "checkpoint_files=0"
    echo "checkpoint_bytes=0"
    echo "checkpoint_human=0B"
    exit 0
fi

python - "${cache_dir}" <<'PY'
import os
import sys

cache_dir = sys.argv[1]
total_bytes = 0
file_count = 0

for root, _, files in os.walk(cache_dir):
    for name in files:
        if name.endswith(".pt"):
            file_count += 1
            total_bytes += os.path.getsize(os.path.join(root, name))

def human_bytes(value):
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if value < 1024:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}EB"

print(f"checkpoint_files={file_count}")
print(f"checkpoint_bytes={total_bytes}")
print(f"checkpoint_human={human_bytes(total_bytes)}")
PY
