#!/usr/bin/env python3
"""Read historical calibrated pilot records; use safe_weighting.py for current runs.

Legacy temperature fields below are read only to label the archived results;
this script does not scale scores or expose a temperature option.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from reporting.core import atomic_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="cifar10_10pct_v1")
    args = parser.parse_args()
    group = STUDY / "runs" / args.group
    rows = []
    for trial in ("baseline_1ep", "three_epochs"):
        config = json.loads((group / trial / "config.json").read_text())
        result = json.loads((group / trial / "result.json").read_text())
        for candidate in result["temperature_candidates"]:
            rows.append(dict(trial=trial, epochs=config["epochs"], pool="scoring", examples=1000,
                prototype_temperature=candidate["temperature"], sisa_accuracy=result["sisa_accuracy"],
                prototype_accuracy=result["prototype_accuracy"], protosisa_accuracy=candidate["accuracy"],
                deleted_examples=0))
    final_path = group / "three_epochs/finalization/result.json"
    if final_path.exists():
        result = json.loads(final_path.read_text())
        for key, pool, deleted in (("confirmation_before", "confirmation", 0),
                                   ("confirmation_after", "confirmation", 1),
                                   ("test_after", "official_test", 1)):
            rows.append(dict(trial="three_epochs", epochs=3, pool=pool,
                prototype_temperature=result["settings"]["prototype_temperature"],
                deleted_examples=deleted, **result[key]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(group / "summary.csv", buffer.getvalue())
    lines = ["# ProtoSISA 10% CIFAR-10 pilot", "",
        "4,750 stratified training examples; ten independent shards; fixed 50/50 score weights.", "",
        "| Configuration | Pool | Examples | SISA | Prototypes | ProtoSISA |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["pool"] == "scoring" and not (
            row["prototype_temperature"] == 1 or (row["trial"] == "three_epochs" and row["prototype_temperature"] == 0.05)):
            continue
        label = f"{row['epochs']} epochs, T={row['prototype_temperature']:g}, {row['deleted_examples']} deleted"
        lines.append(f"| {label} | {row['pool']} | {row['examples']} | "
                     f"{100 * row['sisa_accuracy']:.2f}% | {100 * row['prototype_accuracy']:.2f}% | "
                     f"{100 * row['protosisa_accuracy']:.2f}% |")
    lines += ["", "Settings were selected on scoring examples only. Confirmation and test scores were obtained after freezing settings.",
              "The official test set was evaluated once, after the deletion replay. This pilot does not establish full-data or other-dataset performance.", ""]
    atomic_text(group / "report.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
