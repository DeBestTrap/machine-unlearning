"""Rebuild the shared comparison's tables and plots from saved records."""
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY.parents[1]))

from reporting.study import main
from studies.artifacts import publish_reports

if __name__ == "__main__":
    raise SystemExit(main(STUDY, publish=publish_reports))
