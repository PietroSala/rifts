"""Profile sanity test on the 66 included datasets.

Two claims:
  1. For the trivial ICF of every test sample, every tree's profile is a
     singleton {leaf_idx} whose label matches Def3Forest.per_tree_predict on
     the raw sample.
  2. Widening every feature's interval to (-∞, +∞) produces a profile equal
     to the set of *all* leaves of each tree.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import joblib

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
SCRIPTS = HERE.parents[2] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from drifts.icf import trivial_icf
from drifts.profile import forest_profile
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


REPO = HERE.parents[2]
MODELS = REPO / "models"
ORDER = REPO / "experiments_order" / "included_topo.csv"


def _datasets():
    with ORDER.open() as f:
        return [r["dataset"] for r in csv.DictReader(f) if r.get("dataset")]


def check_one(name: str) -> dict:
    rf = joblib.load(MODELS / f"{name}.joblib")["model"]
    irf = from_sklearn(rf, dataset=name)
    def3 = Def3Forest(rf)
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    ds = load_dataset(name)
    leaves_per_tree = irf.leaves_per_tree()

    # ---- (1) trivial ICF: per-tree singleton, label matches Def3 ----
    X = ds["X_test"]
    n_check = len(X)
    per_tree_def3 = def3.per_tree_predict(X)  # (n_trees, n_samples)
    mismatches = 0
    sizes = []
    for k in range(n_check):
        x = [float(v) for v in X[k]]
        icf = trivial_icf(irf, x)
        prof = forest_profile(irf, icf)
        for j, ps in enumerate(prof):
            sizes.append(len(ps))
            if len(ps) != 1:
                mismatches += 1
                continue
            (leaf_idx,) = ps
            label = leaves_per_tree[j][leaf_idx]["label_idx"]
            if str(inv_phi_L[label]) != str(per_tree_def3[j, k]):
                mismatches += 1
    avg_singleton = sum(sizes) / len(sizes) if sizes else 0.0

    # ---- (2) widened ICF = (-∞, +∞) per feature → profile = all leaves ----
    wide_icf = {i: (-1, len(irf.EU[i])) for i in range(irf.n_features)}
    wide_profile = forest_profile(irf, wide_icf)
    wide_ok = all(len(prof) == len(leaves_per_tree[j])
                  for j, prof in enumerate(wide_profile))

    return {
        "dataset": name, "n_samples": n_check, "n_trees": irf.n_trees,
        "trivial_mismatches": mismatches, "avg_profile_size_trivial": avg_singleton,
        "wide_ok": wide_ok,
    }


def main() -> int:
    datasets = _datasets()
    print(f"profile sanity check over {len(datasets)} datasets\n", flush=True)
    fails = 0
    t0 = time.time()
    for i, name in enumerate(datasets, 1):
        try:
            r = check_one(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:3d}/{len(datasets)}] {name:30s} EXC: {type(exc).__name__}: {exc}",
                  flush=True)
            fails += 1
            continue
        bad = (r["trivial_mismatches"] != 0) or (not r["wide_ok"])
        mark = "OK" if not bad else "FAIL"
        if bad:
            fails += 1
        print(f"[{i:3d}/{len(datasets)}] {r['dataset']:30s} "
              f"n_test={r['n_samples']:>5d} n_trees={r['n_trees']:>4d} "
              f"avg_singleton={r['avg_profile_size_trivial']:.2f} "
              f"wide={'✓' if r['wide_ok'] else '✗'} "
              f"trivial_mm={r['trivial_mismatches']:>4d} {mark}",
              flush=True)
    print(f"\n{len(datasets) - fails}/{len(datasets)} OK ({time.time() - t0:.1f}s)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
