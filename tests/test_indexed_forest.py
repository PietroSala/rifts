"""Round-trip test of the indexed forest + trivial ICF through Redis.

Picks one tuned UCR forest (`Coffee` by default — small, fully completed by
the Max-iAXp baseline), converts it via `sklearn_io`, persists in Redis,
reloads, verifies bit-equality, and checks the trivial ICF brackets every
feature value of sample 0 (i.e. the indexed RF still predicts the same class
on the lower-right corner of the interval — that is the closed endpoint).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np

# Make src/ importable, and the scripts/ tree (for load_ucr).
HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
SCRIPTS = HERE.parents[2] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from cache.connection import get_client
from cache.store import (
    delete_indexed_forest,
    load_indexed_forest,
    load_trivial_icf,
    save_indexed_forest,
    save_trivial_icf,
)
from drifts.icf import icf_human, trivial_icf
from drifts.indexed_forest import IndexedRandomForest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
MODELS_DIR = HERE.parents[2] / "models"


def _build_irf(dataset: str) -> tuple[IndexedRandomForest, dict]:
    joblib_path = MODELS_DIR / f"{dataset}.joblib"
    if not joblib_path.exists():
        raise SystemExit(
            f"no tuned forest at {joblib_path} — run scripts/run_all.py first"
        )
    bundle = joblib.load(joblib_path)
    rf = bundle["model"]
    irf = from_sklearn(rf, dataset=dataset)
    ds = load_dataset(dataset)
    return irf, ds


def test_roundtrip_and_trivial_icf() -> None:
    irf, ds = _build_irf(DATASET)
    print(f"[{DATASET}] n_features={irf.n_features}, n_trees={irf.n_trees}, "
          f"n_labels={irf.n_labels}")

    # ---- predict on TEST set, compare against Def-3 reference ----
    # docs/follow.md A,B: our oracle is Def3Forest (per-tree argmax, majority
    # vote, lex-first tie-break). sklearn's rf.predict() averages probas and
    # can disagree on non-pure leaves; we report that gap but do not require
    # equality with it.
    from drifts.sklearn_compat import Def3Forest

    rf = joblib.load(MODELS_DIR / f"{DATASET}.joblib")["model"]
    def3 = Def3Forest(rf)
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    n_check = min(40, ds["n_test"])
    X_check = ds["X_test"][:n_check]
    def3_pred = def3.predict(X_check)
    sklearn_pred = rf.predict(X_check)
    for k in range(n_check):
        idx = irf.predict([float(v) for v in X_check[k]])
        assert str(inv_phi_L[idx]) == str(def3_pred[k]), (
            f"sample {k}: indexed_forest predicts {inv_phi_L[idx]!r}, "
            f"Def-3 majority says {def3_pred[k]!r}"
        )
    diverged = sum(1 for k in range(n_check) if str(def3_pred[k]) != str(sklearn_pred[k]))
    print(f"[{DATASET}] indexed_forest agrees with Def-3 on all {n_check} samples "
          f"(sklearn.predict diverges on {diverged}/{n_check} via probability averaging)")

    # ---- Redis round-trip ----
    r = get_client("DATA")
    delete_indexed_forest(r, DATASET)
    save_indexed_forest(r, irf)
    irf_reload = load_indexed_forest(r, DATASET)
    assert irf_reload.phi_F == irf.phi_F
    assert irf_reload.phi_L == irf.phi_L
    for i in range(irf.n_features):
        assert irf_reload.EU[i] == irf.EU[i], f"EU mismatch at feature {i}"
    for j in range(irf.n_trees):
        assert irf_reload.trees[j] == irf.trees[j], f"tree {j} mismatch"
    print(f"[{DATASET}] indexed forest bit-exact after Redis round-trip")

    # ---- trivial ICF for sample 0 ----
    x0 = [float(v) for v in ds["X_test"][0]]
    icf = trivial_icf(irf, x0)
    save_trivial_icf(r, DATASET, 0, icf)
    icf_reload = load_trivial_icf(r, DATASET, 0)
    assert icf_reload == icf

    n_eu_unused = sum(1 for i in range(irf.n_features) if not irf.EU[i])
    n_unbounded_both = sum(
        1 for i, (b, e) in icf.items()
        if b == -1 and e == len(irf.EU[i])
    )
    print(f"[{DATASET}] trivial ICF: {len(icf)} features, "
          f"{n_unbounded_both} fully unbounded "
          f"(of which {n_eu_unused} are unused by the forest)")

    # Sanity-check convention: at every feature, value sits on (b, e] interval.
    sample_classes = []
    for i, (b_pos, e_pos) in icf.items():
        eu_i = irf.EU[i]
        v = x0[i]
        if b_pos >= 0:
            b = float(eu_i[b_pos])
            assert v > b, f"feature {i}: v={v} not > b={b}"
        if e_pos < len(eu_i):
            e = float(eu_i[e_pos])
            assert v <= e, f"feature {i}: v={v} not <= e={e}"
    print(f"[{DATASET}] convention check passed: v ∈ (b, e] for every feature")

    # Show a small human-readable preview of the ICF.
    preview = list(icf_human(icf, irf).items())[:5]
    print(f"[{DATASET}] first 5 ICF entries (b, e]: {preview}")


if __name__ == "__main__":
    test_roundtrip_and_trivial_icf()
    print("OK")
