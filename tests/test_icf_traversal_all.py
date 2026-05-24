"""For each of the 66 included datasets, walk every test sample's trivial ICF
through every tree of the indexed forest and compare the leaf labels with
`Def3Forest.per_tree_predict` (which goes through sklearn's `est.apply` on
the raw sample). The two paths must agree on every (dataset, sample, tree)
triple.

This is a cross-dataset correctness check for the Step-1 data structures.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
SCRIPTS = HERE.parents[2] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from drifts.icf import trivial_icf
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn
from load_ucr import load_dataset  # noqa: E402


REPO_ROOT = HERE.parents[2]
MODELS_DIR = REPO_ROOT / "models"
ORDER_CSV = REPO_ROOT / "experiments_order" / "included_topo.csv"


def _read_order() -> list[str]:
    with ORDER_CSV.open() as f:
        return [r["dataset"] for r in csv.DictReader(f) if r.get("dataset")]


def check_one(name: str) -> dict:
    bundle = joblib.load(MODELS_DIR / f"{name}.joblib")
    rf = bundle["model"]
    irf = from_sklearn(rf, dataset=name)
    def3 = Def3Forest(rf)
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    ds = load_dataset(name)
    X_test = ds["X_test"]
    n_samples = len(X_test)

    per_tree_def3 = def3.per_tree_predict(X_test)  # (n_trees, n_samples) of str
    mismatches: list[tuple] = []
    t0 = time.time()
    for k in range(n_samples):
        x = [float(v) for v in X_test[k]]
        icf = trivial_icf(irf, x)
        irf_per_tree = irf.per_tree_labels_from_icf(icf)
        irf_per_tree_str = [str(inv_phi_L[i]) for i in irf_per_tree]
        for j in range(irf.n_trees):
            if irf_per_tree_str[j] != str(per_tree_def3[j, k]):
                mismatches.append((k, j, irf_per_tree_str[j], str(per_tree_def3[j, k])))
    wall = time.time() - t0
    return {
        "dataset": name,
        "n_samples": n_samples,
        "n_trees": irf.n_trees,
        "n_features": irf.n_features,
        "checks": n_samples * irf.n_trees,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "wall_s": round(wall, 2),
    }


def main() -> int:
    datasets = _read_order()
    print(f"checking {len(datasets)} datasets (every sample × every tree)\n", flush=True)
    total_checks = 0
    total_mismatches = 0
    t_global = time.time()
    failed = []
    for i, name in enumerate(datasets, 1):
        try:
            r = check_one(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:3d}/{len(datasets)}] {name:32s} EXCEPTION: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            failed.append((name, repr(exc)))
            continue
        total_checks += r["checks"]
        total_mismatches += r["mismatches"]
        mark = "OK" if r["mismatches"] == 0 else f"MISMATCH ({r['mismatches']})"
        print(f"[{i:3d}/{len(datasets)}] {name:32s} "
              f"n_test={r['n_samples']:>5d} n_trees={r['n_trees']:>4d} "
              f"checks={r['checks']:>7d} {mark} ({r['wall_s']:.1f}s)", flush=True)
        if r["mismatches"]:
            for k, j, a, b in r["mismatch_examples"]:
                print(f"    sample {k} tree {j}: irf={a!r} def3={b!r}", flush=True)

    elapsed = time.time() - t_global
    print(f"\ntotal: {len(datasets)} datasets, {total_checks:,} checks, "
          f"{total_mismatches} mismatches ({elapsed:.1f}s)", flush=True)
    if failed:
        print("failed datasets:")
        for n, e in failed:
            print(f"  {n}: {e}")
    return 0 if (total_mismatches == 0 and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())
