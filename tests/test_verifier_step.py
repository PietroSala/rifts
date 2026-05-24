"""Verifier ``step()`` behaviour — ε query + the absorbed-state short-circuit.

The trivial-ICF on Coffee always lands in absorbing=1 at Init, so we cannot
exercise the ρ-driven branch here without a widened ICF. This test covers:

  * ``step(ε)`` returns the absorbing label after Init absorbed.
  * ``step(0)`` / ``step(1)`` on an already-absorbed verifier returns the
    absorbing label without raising / mutating state.
  * ``reset()`` followed by ``step(ε)`` still returns the same absorbing label.
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
from drifts.verifier import EPSILON, Verifier

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
MODELS = HERE.parents[2] / "models"


def _build_and_init():
    rf = joblib.load(MODELS / f"{DATASET}.joblib")["model"]
    irf = from_sklearn(rf, dataset=DATASET)
    def3 = Def3Forest(rf)
    ds = load_dataset(DATASET)
    ctx = OBDDContext.for_forest(irf, encoding="binary")
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))

    r = get_client("DATA")
    wipe_dataset_caches(r, f"_verifier_step_{DATASET}")
    fwd = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}

    x = [float(v) for v in ds["X_test"][0]]
    icf = trivial_icf(irf, x)
    c_star = label_idx_of[str(fwd[0])]
    caches = open_caches(r, f"_verifier_step_{DATASET}", c_star)
    v = Verifier(irf, ctx, caches, c_star)
    result = v.init(icf)
    assert result == 1, f"trivial-ICF init expected 1, got {result}"
    return v, r


def test_step_epsilon_returns_absorbing():
    v, r = _build_and_init()
    assert v.step(EPSILON) == 1
    assert v.step(None) == 1                  # None is treated as ε
    print(f"[{DATASET}] step(ε) → absorbing OK")
    wipe_dataset_caches(r, f"_verifier_step_{DATASET}")


def test_step_symbol_on_absorbed_short_circuits():
    v, r = _build_and_init()
    # Verifier is already absorbed; step(0) / step(1) should NOT mutate state.
    pre_alpha = v.alpha
    assert v.step(0) == 1
    assert v.step(1) == 1
    assert v.alpha is pre_alpha               # unchanged
    print(f"[{DATASET}] step on absorbed verifier short-circuits OK")
    wipe_dataset_caches(r, f"_verifier_step_{DATASET}")


def test_reset_then_epsilon():
    v, r = _build_and_init()
    v.absorbing = None
    v.reset()
    assert v.step(EPSILON) == 1
    print(f"[{DATASET}] reset then step(ε) → absorbing OK")
    wipe_dataset_caches(r, f"_verifier_step_{DATASET}")


if __name__ == "__main__":
    test_step_epsilon_returns_absorbing()
    test_step_symbol_on_absorbed_short_circuits()
    test_reset_then_epsilon()
    print("OK")
