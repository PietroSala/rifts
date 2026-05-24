"""Sweep `train_baseline.tune_dataset` across all UCR univariate datasets.

Writes one row per dataset to metrics/baseline.csv; resumable: datasets already
present in the CSV are skipped unless --force is set.
"""
from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

from load_ucr import list_datasets
from train_baseline import N_TRIALS, TIMEOUT_SECONDS, tune_dataset

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
METRICS_CSV = REPO_ROOT / "metrics" / "baseline.csv"

FIELDS = [
    "dataset",
    "n_train",
    "n_test",
    "length",
    "n_classes",
    "best_params",
    "cv_acc",
    "test_acc",
    "n_trials_completed",
    "wall_s",
    "model_path",
]


def already_done(names: set[str]) -> set[str]:
    if not METRICS_CSV.exists():
        return set()
    with METRICS_CSV.open() as f:
        reader = csv.DictReader(f)
        return {row["dataset"] for row in reader if row["dataset"] in names}


def append_row(row: dict) -> None:
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    parser.add_argument("--force", action="store_true", help="re-tune even if already in CSV")
    args = parser.parse_args()

    datasets = list_datasets()
    if args.only:
        datasets = [d for d in datasets if d in set(args.only)]
    done = set() if args.force else already_done(set(datasets))

    t0 = time.time()
    for i, name in enumerate(datasets, 1):
        if name in done:
            print(f"[{i:3d}/{len(datasets)}] {name:30s} SKIP (already in CSV)", flush=True)
            continue
        print(f"[{i:3d}/{len(datasets)}] {name:30s} tuning…", flush=True)
        try:
            row = tune_dataset(name, n_trials=args.n_trials, timeout=args.timeout)
            append_row(row)
            print(
                f"    cv_acc={row['cv_acc']:.4f} test_acc={row['test_acc']:.4f} "
                f"trials={row['n_trials_completed']} wall={row['wall_s']:.1f}s",
                flush=True,
            )
        except Exception:
            print(f"    FAILED on {name}", flush=True)
            traceback.print_exc()
    print(f"\nTotal wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
