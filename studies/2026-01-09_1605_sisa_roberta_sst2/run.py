#!/usr/bin/env python3
"""Launch this study's SISA recipes; resolved settings are captured per run."""
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))
from studies.sisa import main

if __name__ == "__main__":
    raise SystemExit(main(STUDY))
