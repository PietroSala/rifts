"""Running OBDD D + derived δ — correctness checks.

Two claims:

  1. With α built from the trivial-ICF profile (every tree's reachable
     set is the singleton `{v*}`; α forbids every other leaf), the
     running OBDD `D` is satisfiable AND the derived δ marks `v*` as
     forced to 1 in every tree.

  2. Manually marking every leaf of one tree as 0 in α makes the
     corridor structurally empty: `is_corridor_unsat(D)` returns True.
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

from drifts.icf import trivial_icf
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.partial_assignment import PartialAssignment
from drifts.profile import forest_profile
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
ENCODING = os.environ.get("DRIFTS_TEST_ENCODING", "binary")
N_SAMPLES = int(os.environ.get("DRIFTS_TEST_N_SAMPLES", "5"))
MODELS = HERE.parents[2] / "models"


def _build(dataset: str, encoding: str):
    rf = joblib.load(MODELS / f"{dataset}.joblib")["model"]
    irf = from_sklearn(rf, dataset=dataset)
    ctx = OBDDContext.for_forest(irf, encoding=encoding)
    t0 = time.time()
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    print(f"[{dataset}/{encoding}] bootstrap: {time.time() - t0:.2f}s "
          f"(n_vars={ctx.enc.n_variables()}, "
          f"leaves={sum(len(m) for m in ctx.leaf_formulas)})", flush=True)
    return irf, ctx


def test_trivial_icf_forces_reached_leaves():
    irf, ctx = _build(DATASET, ENCODING)
    ds = load_dataset(DATASET)
    n_leaves_per_tree = irf.n_leaves_per_tree()
    n_check = min(N_SAMPLES, ds["n_test"])

    total_trees = 0
    for k in range(n_check):
        x = [float(v) for v in ds["X_test"][k]]
        icf = trivial_icf(irf, x)
        profile = forest_profile(irf, icf)
        # every per-tree profile should be a singleton on the trivial ICF
        for t, prof_t in enumerate(profile):
            assert len(prof_t) == 1, (
                f"sample {k} tree {t}: profile has {len(prof_t)} leaves, "
                f"expected singleton"
            )

        alpha = PartialAssignment.initial_from_profile(profile, n_leaves_per_tree)
        t0 = time.time()
        D = ctx.build_D(alpha)
        t_build = time.time() - t0
        assert not ctx.is_corridor_unsat(D), (
            f"sample {k}: corridor should be non-empty"
        )

        t0 = time.time()
        delta = ctx.compute_delta(alpha, D)
        t_delta = time.time() - t0

        # for every tree, the singleton v_star must be forced to 1 — either
        # directly by the canonical α (since the singleton profile auto-promotes
        # to a 1-mark) or via δ.
        for t, prof_t in enumerate(profile):
            (v_star,) = prof_t
            got = alpha.value(t, v_star)
            if got is None:
                got = delta.value(t, v_star)
            assert got == 1, (
                f"sample {k} tree {t}: (α ∪ δ)({v_star}) = {got!r}, expected 1"
            )
            total_trees += 1

        if k < 2 or k == n_check - 1:
            print(f"[{DATASET}/{ENCODING}] sample {k:>3d}: "
                  f"build_D {t_build*1000:.1f}ms, "
                  f"compute_delta {t_delta*1000:.1f}ms, "
                  f"|α|={alpha.n_decided()}, |δ|={delta.n_decided()}", flush=True)

    print(f"[{DATASET}/{ENCODING}] {n_check} samples × {irf.n_trees} trees "
          f"= {total_trees} tree-checks OK (every v⋆ forced by δ)",
          flush=True)


def test_extinguished_tree_makes_corridor_unsat():
    irf, ctx = _build(DATASET, ENCODING)
    n_leaves_per_tree = irf.n_leaves_per_tree()
    # Mark every leaf of tree 0 as forbidden — corridor must be empty.
    # Bulk construction so canonicalisation does not auto-promote at the
    # `n_t − 1` 0-mark threshold.
    alpha = PartialAssignment(
        decided={0: {v: 0 for v in range(n_leaves_per_tree[0])}},
        n_leaves_per_tree=n_leaves_per_tree,
    )
    D = ctx.build_D(alpha)
    assert ctx.is_corridor_unsat(D), (
        "α forbidding every leaf of tree 0 should make the corridor empty"
    )
    print(f"[{DATASET}/{ENCODING}] extinguished-tree corridor → ⊥ OK",
          flush=True)


if __name__ == "__main__":
    test_trivial_icf_forces_reached_leaves()
    test_extinguished_tree_makes_corridor_unsat()
    print("OK")
