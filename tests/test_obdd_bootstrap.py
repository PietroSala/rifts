"""Bootstrap modes for OBDDContext: per_worker and shared (Redis DDDMP).

  1. `per_worker` mode: build everything locally; verify the existing
     leaf-formula sanity (Ψ(v) selects exactly one leaf per tree on the
     sample's cell tuple).
  2. `shared` mode: two contexts pointing at the same dataset.
     - Worker A: ready_key absent → acquires builder lock → builds + dumps.
     - Worker B: ready_key present → loads from Redis.
     - Verify B's leaf formulas agree with A's on the sample's cell tuple.
     - Verify B's build was faster than a from-scratch local build.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import joblib

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
SCRIPTS = HERE.parents[2] / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import redis as _redis

from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
MODELS_DIR = HERE.parents[2] / "models"
N_SAMPLES = int(os.environ.get("DRIFTS_TEST_N_SAMPLES", "3"))


def _redis_client() -> _redis.Redis:
    # decode_responses=False to keep the DDDMP base64 blob untouched.
    return _redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                        port=int(os.environ.get("REDIS_PORT", 6379)),
                        db=0, decode_responses=False)


def _wipe(r, ds: str) -> None:
    for enc in ("one_hot", "binary"):
        for sfx in ("blob", "meta", "ready", "builder_lock"):
            r.delete(f"{ds}:OBDD:{enc}:{sfx}")


def _check_leaf_formulas(ctx: OBDDContext, irf, ds, def3) -> int:
    """Return # mismatches across N_SAMPLES × n_trees."""
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    leaves_per_tree = irf.leaves_per_tree()
    X = ds["X_test"][:N_SAMPLES]
    per_tree_def3 = def3.per_tree_predict(X)
    mm = 0
    for k in range(len(X)):
        x = [float(v) for v in X[k]]
        sub = ctx.assignment_from_cells(ctx.sample_cell_tuple(x))
        for j, tree_map in enumerate(ctx.leaf_formulas):
            target = str(per_tree_def3[j, k])
            reached = None
            for leaf_idx, psi in tree_map.items():
                if ctx.bdd.let(sub, psi) == ctx.bdd.true:
                    if reached is not None:
                        mm += 1
                    reached = leaf_idx
            if reached is None:
                mm += 1
                continue
            got = str(inv_phi_L[leaves_per_tree[j][reached]["label_idx"]])
            if got != target:
                mm += 1
    return mm


def _run_per_worker(encoding: str) -> None:
    rf = joblib.load(MODELS_DIR / f"{DATASET}.joblib")["model"]
    irf = from_sklearn(rf, dataset=DATASET)
    def3 = Def3Forest(rf)
    ds = load_dataset(DATASET)

    ctx = OBDDContext.for_forest(irf, encoding=encoding)
    t0 = time.time()
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    n_vars = ctx.enc.n_variables()
    print(f"[{DATASET}/{encoding}] per_worker bootstrap: "
          f"{time.time() - t0:.2f}s (n_vars={n_vars}, "
          f"leaves={sum(len(m) for m in ctx.leaf_formulas)})",
          flush=True)
    assert ctx.is_bootstrapped
    mm = _check_leaf_formulas(ctx, irf, ds, def3)
    assert mm == 0, f"per_worker {encoding} leaf formula mismatches: {mm}"
    print(f"[{DATASET}/{encoding}] per_worker check OK (0 mismatches)",
          flush=True)


def _run_shared(encoding: str) -> None:
    rf = joblib.load(MODELS_DIR / f"{DATASET}.joblib")["model"]
    irf = from_sklearn(rf, dataset=DATASET)
    def3 = Def3Forest(rf)
    ds = load_dataset(DATASET)

    r = _redis_client()
    _wipe(r, DATASET)

    ctx_a = OBDDContext.for_forest(irf, encoding=encoding)
    t0 = time.time()
    ctx_a.bootstrap(BootstrapConfig(mode="shared", redis=r, dataset=DATASET))
    t_build = time.time() - t0
    print(f"[{DATASET}/{encoding}] worker A (shared, cold): {t_build:.2f}s",
          flush=True)
    assert r.get(f"{DATASET}:OBDD:{encoding}:ready") == b"1"
    assert r.get(f"{DATASET}:OBDD:{encoding}:blob") is not None
    mm_a = _check_leaf_formulas(ctx_a, irf, ds, def3)
    assert mm_a == 0, f"worker-A {encoding} leaf formula mismatches: {mm_a}"

    ctx_b = OBDDContext.for_forest(irf, encoding=encoding)
    t0 = time.time()
    ctx_b.bootstrap(BootstrapConfig(mode="shared", redis=r, dataset=DATASET))
    t_load = time.time() - t0
    print(f"[{DATASET}/{encoding}] worker B (shared, warm load): "
          f"{t_load:.2f}s (speedup ×{t_build / max(t_load, 0.001):.1f})",
          flush=True)
    assert ctx_b.is_bootstrapped
    mm_b = _check_leaf_formulas(ctx_b, irf, ds, def3)
    assert mm_b == 0, f"worker-B {encoding} leaf formula mismatches: {mm_b}"
    print(f"[{DATASET}/{encoding}] shared bootstrap check OK", flush=True)

    _wipe(r, DATASET)


def test_per_worker_one_hot():    _run_per_worker("one_hot")
def test_per_worker_binary():     _run_per_worker("binary")
def test_shared_one_hot():        _run_shared("one_hot")
def test_shared_binary():         _run_shared("binary")


if __name__ == "__main__":
    test_per_worker_one_hot()
    test_per_worker_binary()
    test_shared_one_hot()
    test_shared_binary()
    print("OK")
