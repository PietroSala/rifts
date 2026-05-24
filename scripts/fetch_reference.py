"""Fetch published reference accuracies from tsml-eval into metrics/reference.csv.

Two references are recorded per dataset:
  - acc_tsf    : Time Series Forest mean accuracy across 30 resamples. Closest
                 cousin of our point-feature RF (also an RF, but with interval-
                 derived features).
  - acc_1nn_dtw: 1-NN with Dynamic Time Warping mean accuracy across 30 resamples.
                 The canonical TS lower-bar baseline.

Source: github.com/time-series-machine-learning/tsml-eval/results/classification/Univariate.
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
OUT = REPO_ROOT / "metrics" / "reference.csv"

SOURCES = {
    "acc_tsf": (
        "https://raw.githubusercontent.com/time-series-machine-learning/"
        "tsml-eval/main/results/classification/Univariate/TSF_accuracy.csv"
    ),
    "acc_1nn_dtw": (
        "https://raw.githubusercontent.com/time-series-machine-learning/"
        "tsml-eval/main/results/classification/Univariate/1NN-DTW_accuracy.csv"
    ),
}


def fetch_table(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url) as r:
        text = r.read().decode("utf-8")
    rows = [line.split(",") for line in text.strip().splitlines()]
    out: dict[str, float] = {}
    for row in rows[1:]:
        name = row[0].strip()
        vals = [float(v) for v in row[1:] if v.strip()]
        if vals:
            out[name] = sum(vals) / len(vals)
    return out


def main() -> None:
    tables = {k: fetch_table(url) for k, url in SOURCES.items()}
    names = sorted(set().union(*[t.keys() for t in tables.values()]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "acc_tsf", "acc_1nn_dtw", "source"])
        for name in names:
            writer.writerow([
                name,
                f"{tables['acc_tsf'].get(name, ''):.6f}" if name in tables["acc_tsf"] else "",
                f"{tables['acc_1nn_dtw'].get(name, ''):.6f}" if name in tables["acc_1nn_dtw"] else "",
                "tsml-eval (mean over resamples)",
            ])

    n_tsf = sum(1 for n in names if n in tables["acc_tsf"])
    n_dtw = sum(1 for n in names if n in tables["acc_1nn_dtw"])
    print(f"wrote {OUT} ({len(names)} datasets; TSF={n_tsf}, 1NN-DTW={n_dtw})")


if __name__ == "__main__":
    main()
