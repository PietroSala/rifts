"""Compute per-dataset model + Endpoint-Universe statistics from the tuned RFs.

Writes metrics/model_stats.json with one entry per dataset:
  - n_test          : size of the test split
  - length          : series length (== n_features)
  - n_classes       : number of classes
  - total_leaves    : sum over trees of estimator.tree_.n_leaves
  - n_unused_features: count of features that no tree ever splits on
  - eu_per_feature  : list of |EU(i)| for i = 0..length-1 (raw data)
  - eu_stats        : {mean, min, max, std} over eu_per_feature
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from load_ucr import load_dataset  # noqa: E402
from max_iaxp.coverage import extract_eu  # noqa: E402

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
MODELS_DIR = REPO_ROOT / "models"
OUT = REPO_ROOT / "metrics" / "model_stats.json"


def stats_for(name: str) -> dict | None:
    model_path = MODELS_DIR / f"{name}.joblib"
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    rf = bundle["model"]
    n_features = int(rf.n_features_in_)
    eu = extract_eu(rf)
    sizes = [len(eu[i]) for i in range(n_features)]
    leaves_per_tree = [int(est.tree_.n_leaves) for est in rf.estimators_]
    total_leaves = sum(leaves_per_tree)
    n_unused = sum(1 for s in sizes if s == 0)
    used_sizes = [s for s in sizes if s > 0]
    ds = load_dataset(name)
    # EU statistics (μ, min, max, σ) are computed over **used** features only
    # (those with at least one split threshold). Unused features are already
    # surfaced separately as n_unused_features; including them in the average
    # / spread just dilutes the signal with zeros. `max` over used == max over
    # all (unused contribute 0), so it is unaffected.
    return {
        "n_test": int(ds["n_test"]),
        "length": n_features,
        "n_classes": int(ds["n_classes"]),
        "n_trees": int(rf.n_estimators),
        "max_depth": int(rf.max_depth) if rf.max_depth is not None else None,
        "total_leaves": total_leaves,
        "n_unused_features": n_unused,
        "eu_per_feature": sizes,
        "eu_stats": {
            "mean": float(np.mean(used_sizes)) if used_sizes else 0.0,
            "min": int(np.min(used_sizes)) if used_sizes else 0,
            "max": int(np.max(used_sizes)) if used_sizes else 0,
            "std": float(np.std(used_sizes)) if used_sizes else 0.0,
        },
        "leaves_per_tree": leaves_per_tree,
        "leaves_per_tree_stats": {
            "mean": float(np.mean(leaves_per_tree)),
            "min": int(np.min(leaves_per_tree)),
            "max": int(np.max(leaves_per_tree)),
            "std": float(np.std(leaves_per_tree)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    datasets = sorted(p.stem for p in MODELS_DIR.glob("*.joblib"))
    if args.only:
        datasets = [d for d in datasets if d in set(args.only)]

    out: dict[str, dict] = {}
    t0 = time.time()
    for i, name in enumerate(datasets, 1):
        s = stats_for(name)
        if s is None:
            print(f"  skip {name}: no joblib", flush=True)
            continue
        out[name] = s
        if i % 10 == 0 or i <= 3 or i == len(datasets):
            es = s["eu_stats"]
            print(
                f"[{i:3d}/{len(datasets)}] {name:30s} "
                f"leaves={s['total_leaves']:>5d} unused={s['n_unused_features']:>3d} "
                f"eu mean={es['mean']:.1f} min={es['min']} max={es['max']} std={es['std']:.1f}",
                flush=True,
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT} ({len(out)} datasets) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
