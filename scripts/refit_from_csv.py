"""Deterministically refit one forest per dataset from metrics/baseline.csv.

This is the reproducibility entry point: given the committed `baseline.csv`,
which records the best hyperparameters found by Optuna for each UCR dataset,
this script refits each forest with the same seed and verifies that the test
accuracy matches the value recorded in the CSV. No joblib artefacts need to
be shipped — the CSV plus this script reconstructs every model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier

from load_ucr import load_dataset

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
METRICS_CSV = REPO_ROOT / "metrics" / "baseline.csv"
MODELS_DIR = REPO_ROOT / "models"

SEED = 0
EXPECTED_SKLEARN = "1.4"
ACC_TOLERANCE = 1e-9


def read_rows() -> list[dict]:
    if not METRICS_CSV.exists():
        sys.exit(f"missing {METRICS_CSV}. Run scripts/run_all.py first.")
    with METRICS_CSV.open() as f:
        return list(csv.DictReader(f))


def refit_one(row: dict, save: bool = True, verify: bool = True) -> dict:
    name = row["dataset"]
    params = json.loads(row["best_params"])
    final_params = {
        **params,
        "criterion": "gini",
        "random_state": SEED,
        "n_jobs": -1,
    }
    ds = load_dataset(name)
    clf = RandomForestClassifier(**final_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(ds["X_train"], ds["y_train"])
    test_acc = float(clf.score(ds["X_test"], ds["y_test"]))

    expected_acc = float(row["test_acc"])
    ok = math.isclose(test_acc, expected_acc, abs_tol=ACC_TOLERANCE)
    if verify and not ok:
        print(
            f"  WARN  {name}: refit test_acc={test_acc:.6f} != CSV {expected_acc:.6f} "
            f"(Δ={test_acc - expected_acc:+.6f})"
        )

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": clf, "best_params": final_params, "dataset": name},
            MODELS_DIR / f"{name}.joblib",
            compress=3,
        )

    return {"dataset": name, "expected": expected_acc, "refit": test_acc, "match": ok}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    parser.add_argument("--no-save", action="store_true", help="do not write joblibs")
    parser.add_argument("--no-verify", action="store_true", help="skip accuracy verification")
    args = parser.parse_args()

    if not sklearn.__version__.startswith(EXPECTED_SKLEARN):
        print(
            f"WARN: sklearn {sklearn.__version__} != expected {EXPECTED_SKLEARN}.x; "
            "refits may differ from the CSV",
            file=sys.stderr,
        )

    rows = read_rows()
    if args.only:
        wanted = set(args.only)
        rows = [r for r in rows if r["dataset"] in wanted]

    mismatches = 0
    for i, row in enumerate(rows, 1):
        result = refit_one(row, save=not args.no_save, verify=not args.no_verify)
        mark = "OK" if result["match"] else "MISMATCH"
        print(
            f"[{i:3d}/{len(rows)}] {result['dataset']:30s} "
            f"refit={result['refit']:.4f} csv={result['expected']:.4f} {mark}",
            flush=True,
        )
        if not result["match"]:
            mismatches += 1

    print(f"\n{len(rows) - mismatches}/{len(rows)} datasets match the CSV.")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
