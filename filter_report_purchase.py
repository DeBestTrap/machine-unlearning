#!/usr/bin/env python3
import argparse
import json

import pandas as pd


def load_hyperparams(series):
    return series.apply(lambda s: json.loads(s) if isinstance(s, str) else {})


def changing_keys(hparams):
    keys = set()
    for item in hparams:
        keys.update(item.keys())

    changing = []
    for key in sorted(keys):
        values = {item.get(key) for item in hparams}
        if len(values) > 1:
            changing.append(key)
    return changing


def main():
    parser = argparse.ArgumentParser(
        description="Print a DataFrame with only hyperparams that change across rows."
    )
    parser.add_argument(
        "--csv",
        default="report-purchase.csv",
        help="Path to the input report CSV (default: report-purchase.csv).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    hparams = load_hyperparams(df["hyperparams_json"])
    keys = changing_keys(hparams)

    out = df.drop(columns=["hyperparams_json"])
    for key in keys:
        out[key] = hparams.apply(lambda d, k=key: d.get(k))

    print(out)


if __name__ == "__main__":
    main()
