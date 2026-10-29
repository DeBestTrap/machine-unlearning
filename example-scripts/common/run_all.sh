#!/bin/bash

set -eou pipefail
IFS=$'\n\t'

if [[ $# -lt 13 ]]; then
    echo "Usage: $0 <model> <dataset_path> <shards> <ul_reqs> <container_name> <slices> <epochs> <batch_size> <learning_rate> <optimizer> <chkpt_interval> <predict_batch_size> <report_name> [dropout_rate]"
    exit 1
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
python - "${container_name}" "${model}" "${dataset_path}" "${shards}" "${ul_reqs}" "${hparams}" "${predict_batch_size}" "${dropout_rate}" <<'PY'
import json
import os
from pathlib import Path
import sys
import uuid
import torch

from container_storage import record_directory
from reporting.core import hardware, runtime_provenance

container, model, dataset, shards, steps, hparams, predict_batch_size, dropout = sys.argv[1:]
config = {
    "comparison_run_id": "sisa-" + uuid.uuid4().hex,
    "seed": int(os.environ["SISA_SEED"]) if "SISA_SEED" in os.environ else None,
    "provenance": runtime_provenance(Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()),
    "hardware": hardware(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
    "model": model,
    "dataset": dataset,
    "dataset_metadata": json.loads(Path(dataset).read_text()),
    "shards": int(shards),
    "deletion_steps": int(steps),
    "hyperparams": json.loads(hparams),
    "predict_batch_size": int(predict_batch_size),
    "dropout_rate": float(dropout) if dropout else 0.4,
}
path = record_directory(container) / "config.json"
if path.exists():
    previous = json.loads(path.read_text())
    identity = set(config) - {"comparison_run_id", "provenance", "hardware"}
    if any(previous.get(key) != config[key] for key in identity):
        raise ValueError("Saved SISA configuration differs; use a fresh run directory")
else:
    path.write_text(json.dumps(config, indent=2) + "\n")
PY
bash "${script_dir}/train.sh" "${model}" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}" "${slices}" "${epochs}" "${batch_size}" "${learning_rate}" "${optimizer}" "${chkpt_interval}" "${dropout_rate}"
bash "${script_dir}/checkpoints_size.sh" "${container_name}"
bash "${script_dir}/predict.sh" "${model}" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}" "${predict_batch_size}"
bash "${script_dir}/data.sh" "${dataset_path}" "${shards}" "${ul_reqs}" "${container_name}" "${report_name}" "${hparams}"
python -m reporting refresh "${container_name}" --output "${report_name}-comparison"
