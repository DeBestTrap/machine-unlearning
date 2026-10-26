#!/usr/bin/env bash
set -euo pipefail

CONTAINERS_DIR="./containers"

if [[ ! -d "${CONTAINERS_DIR}" ]]; then
  echo "containers directory not found: ${CONTAINERS_DIR}" >&2
  exit 1
fi

# Remove contents but keep the containers directory itself.
rm -rf "${CONTAINERS_DIR:?}"/*

echo "Cleared: ${CONTAINERS_DIR}"
