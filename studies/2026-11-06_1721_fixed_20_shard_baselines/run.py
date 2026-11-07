#!/usr/bin/env python3
"""Dispatch the fixed-20 SISA/ProtoSISA or S3T study driver."""
import os
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
ENGINES = {"baseline": STUDY / "baseline.py", "s3t": STUDY / "s3t.py"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENGINES:
        raise SystemExit("Usage: run.py {baseline|s3t} <driver arguments>\n"
                         "Use 'run.py baseline --help' or 'run.py s3t --help'.")
    os.execv(sys.executable, [sys.executable, str(ENGINES[sys.argv[1]]), *sys.argv[2:]])


if __name__ == "__main__":
    main()
