"""Verifier Init smoke tests on Coffee.

Three claims, all on the trivial ICF (every per-tree profile is a singleton
→ canonical α is already complete after ``initial_from_profile``):

  1. With ``c⋆ = Def3.predict(x)``, ``Init`` lands in ``absorbing = 1`` via the
     trivial-slackless Good branch (or via G-cache hit on a previous sample
     within the same dataset / c⋆).
  2. With ``c⋆ ≠ Def3.predict(x)`` and at least two labels, ``Init`` raises
     ``BadException`` via the trivial-slackless Bad branch (or B-cache hit).
  3. ``reset()`` after Init restores the same absorbing label.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
SCRIPTS = HERE.parents[2] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from cache.caches import open_caches, wipe_dataset_caches
from cache.connection import get_client
from drifts.icf import trivial_icf
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn
from drifts.verifier import BadException, Verifier

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
N_SAMPLES = int(os.environ.get("DRIFTS_TEST_N_SAMPLES", "5"))
MODELS = HERE.parents[2] / "models"


def _build():
    rf = joblib.load(MODELS / f"{DATASET}.joblib")["model"]
    irf = from_sklearn(rf, dataset=DATASET)
    def3 = Def3Forest(rf)
    ds = load_dataset(DATASET)
    ctx = OBDDContext.for_forest(irf, encoding="binary")
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    return rf, irf, def3, ds, ctx


def test_verifier_init_trivial_icf_correct_seed_is_good():
    r = get_client("DATA")
    wipe_dataset_caches(r, f"_verifier_{DATASET}")
    rf, irf, def3, ds, ctx = _build()
    fwd_pred = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}

    n_check = min(N_SAMPLES, ds["n_test"])
    n_good = 0
    for k in range(n_check):
        x = [float(v) for v in ds["X_test"][k]]
        icf = trivial_icf(irf, x)
        c_star = label_idx_of[str(fwd_pred[k])]
        caches = open_caches(r, f"_verifier_{DATASET}", c_star)
        v = Verifier(irf, ctx, caches, c_star)
        result = v.init(icf)
        assert result == 1, (
            f"sample {k}: expected absorbing=1 (Good), got {result}; "
            f"absorbing={v.absorbing}, complete={v.alpha.is_complete(irf.n_leaves_per_tree())}"
        )
        n_good += 1
    print(f"[{DATASET}] {n_good}/{n_check} trivial-ICF Init → absorbing=1 OK")
    wipe_dataset_caches(r, f"_verifier_{DATASET}")


def test_verifier_init_trivial_icf_wrong_seed_raises_bad():
    r = get_client("DATA")
    wipe_dataset_caches(r, f"_verifier_{DATASET}")
    rf, irf, def3, ds, ctx = _build()
    if irf.n_labels < 2:
        print(f"[{DATASET}] single-label dataset, skipping wrong-seed test")
        return
    fwd_pred = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}

    n_check = min(N_SAMPLES, ds["n_test"])
    n_bad = 0
    for k in range(n_check):
        x = [float(v) for v in ds["X_test"][k]]
        icf = trivial_icf(irf, x)
        c_star = label_idx_of[str(fwd_pred[k])]
        wrong = next(c for c in range(irf.n_labels) if c != c_star)
        caches = open_caches(r, f"_verifier_{DATASET}", wrong)
        v = Verifier(irf, ctx, caches, wrong)
        try:
            v.init(icf)
            assert False, f"sample {k}: wrong seed should raise BadException"
        except BadException as exc:
            assert exc.alpha is not None
            n_bad += 1
    print(f"[{DATASET}] {n_bad}/{n_check} trivial-ICF wrong-seed → BadException OK")
    wipe_dataset_caches(r, f"_verifier_{DATASET}")


def test_verifier_reset_restores_absorbing():
    r = get_client("DATA")
    wipe_dataset_caches(r, f"_verifier_{DATASET}")
    rf, irf, def3, ds, ctx = _build()
    fwd_pred = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}

    x = [float(v) for v in ds["X_test"][0]]
    icf = trivial_icf(irf, x)
    c_star = label_idx_of[str(fwd_pred[0])]
    caches = open_caches(r, f"_verifier_{DATASET}", c_star)
    v = Verifier(irf, ctx, caches, c_star)
    a0 = v.init(icf)
    # Mutate state, then reset
    v.absorbing = None
    v.alpha = None
    v.reset()
    assert v.absorbing == a0
    print(f"[{DATASET}] reset() restores absorbing={a0} OK")
    wipe_dataset_caches(r, f"_verifier_{DATASET}")


if __name__ == "__main__":
    test_verifier_init_trivial_icf_correct_seed_is_good()
    test_verifier_init_trivial_icf_wrong_seed_raises_bad()
    test_verifier_reset_restores_absorbing()
    print("OK")
