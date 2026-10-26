#!/usr/bin/env bash
set -euo pipefail

# Keep timings, predictions, requests, manifests, and reports for later analysis.
python container_storage.py clear-models "$@"
