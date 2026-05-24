"""Exercise the Continue branch of Verifier.Init/Step.

Strategy: start from a trivial ICF (slackless, absorbs at Init), then
progressively WIDEN feature ranges by extending ``(b_pos, e_pos)`` until the
resulting ICF's profile is rich enough that Init lands in the Continue
state (returns ``None``, ``absorbing is None``, ``rho`` non-empty).

Once in Continue, drive ``step(0/1)`` with the *real-sample* value at each
ρ-head leaf — that gives a sound and terminating sequence: every step either
advances or absorbs, and the final absorbing label is ``1`` (the sample is
genuinely in c⋆'s corridor under the widened ICF, since c⋆ = Def3(x)).

We also confirm:
  * ``step(ε)`` after termination returns the absorbing label.
  * The full forward walk ends with ``rho`` empty AND ``absorbing == 1``.
"""
from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

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
from drifts.profile import forest_profile
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn
from drifts.verifier import EPSILON, BadException, Verifier

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
MODELS = HERE.parents[2] / "models"
SEED = int(os.environ.get("DRIFTS_TEST_SEED", "0"))


def _widen(icf: Dict[int, Tuple[int, int]], irf, features, delta: int):
    """Return a widened copy: extend (b_pos, e_pos) by ``delta`` on each side
    for the given feature indices, clamped to the valid EU range."""
    out = dict(icf)
    for f in features:
        eu_n = len(irf.EU[f])
        b, e = icf[f]
        out[f] = (max(-1, b - delta), min(eu_n, e + delta))
    return out


def _leaf_value_on_sample(ctx, irf, x, tree_idx: int, leaf_idx: int) -> int:
    """Return the 0/1 value the *real* sample ``x`` gives to leaf
    ``(tree_idx, leaf_idx)`` — i.e. whether the sample reaches that leaf in
    the corresponding sklearn tree.
    """
    cell_tuple = ctx.sample_cell_tuple(x)
    sub = ctx.assignment_from_cells(cell_tuple)
    psi = ctx.leaf_formulas[tree_idx][leaf_idx]
    return 1 if ctx.bdd.let(sub, psi) == ctx.bdd.true else 0


def _build():
    rf = joblib.load(MODELS / f"{DATASET}.joblib")["model"]
    irf = from_sklearn(rf, dataset=DATASET)
    def3 = Def3Forest(rf)
    ds = load_dataset(DATASET)
    ctx = OBDDContext.for_forest(irf, encoding="binary")
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    return rf, irf, def3, ds, ctx


def _find_continue_icf(irf, ctx, def3, ds, sample_idx: int):
    """Widen the trivial ICF uniformly until Init lands in Continue.

    Tries a small ladder of ``delta`` values (uniform over all features).
    Stops at the first widening whose profile is non-trivially slackless AND
    yields a Continue outcome from Init. Uses a fresh cache prefix per
    attempt — no Redis SCAN-based wipes inside the search loop.
    """
    x = [float(v) for v in ds["X_test"][sample_idx]]
    fwd_pred = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}
    c_star = label_idx_of[str(fwd_pred[sample_idx])]

    trivial = trivial_icf(irf, x)
    r = get_client("DATA")
    all_features = list(range(irf.n_features))
    # Per-process session id to keep cache prefixes fresh across reruns
    # without paying for SCAN-based wipes.
    session = int(time.time() * 1000) ^ os.getpid()

    for attempt, delta in enumerate((1, 2, 3, 4)):
        icf = _widen(trivial, irf, all_features, delta)
        prof = forest_profile(irf, icf)
        if all(len(p) == 1 for p in prof):
            continue  # widening too small, no slack introduced

        ds_key = f"_verif_cont_{DATASET}_{sample_idx}_{session}_d{delta}_a{attempt}"
        caches = open_caches(r, ds_key, c_star)
        v = Verifier(irf, ctx, caches, c_star,
                     rng=random.Random(SEED + sample_idx))
        try:
            outcome = v.init(icf)
        except BadException:
            continue
        if outcome is None and v.absorbing is None and v.rho:
            return icf, x, c_star, v, ds_key, r

    raise RuntimeError(
        f"sample {sample_idx}: no widening in delta ∈ {{1..4}} reached Continue"
    )


def test_init_reaches_continue_with_widened_icf():
    rf, irf, def3, ds, ctx = _build()
    icf, x, c_star, v, ds_key, r = _find_continue_icf(irf, ctx, def3, ds, sample_idx=0)

    # Snapshot sanity
    assert v.snapshot is not None
    assert v.snapshot.absorbing is None
    assert len(v.snapshot.rho) > 0
    assert v.snapshot.D is not None

    # ε on the un-absorbed verifier returns 0 (per the DOT)
    assert v.step(EPSILON) == 0

    print(f"[{DATASET}] sample 0 reached Continue: |α.decided|={v.alpha.n_decided()}, "
          f"|ρ|={len(v.rho)} undecided leaves")

    wipe_dataset_caches(r, ds_key)


def test_step_forward_walk_to_absorption():
    """Drive Step with the real-sample bit at each ρ-head leaf. The walk must
    terminate in ``absorbing == 1`` (the sample is in c⋆'s corridor)."""
    rf, irf, def3, ds, ctx = _build()
    icf, x, c_star, v, ds_key, r = _find_continue_icf(irf, ctx, def3, ds, sample_idx=0)

    n_steps = 0
    max_steps = 10 * v.irf.n_trees * max(v.irf.n_leaves_per_tree())
    while v.absorbing is None:
        assert v.rho, "ρ empty without absorbing"
        h_tree, h_leaf = v.rho[0]
        sigma = _leaf_value_on_sample(ctx, irf, x, h_tree, h_leaf)
        v.step(sigma)
        n_steps += 1
        assert n_steps < max_steps, f"too many steps ({n_steps})"

    assert v.absorbing == 1, (
        f"sample 0: forward walk with the real sample's bits absorbed to "
        f"{v.absorbing}, expected 1"
    )
    assert v.step(EPSILON) == 1
    print(f"[{DATASET}] sample 0 forward walk: {n_steps} steps → absorbing=1 OK")

    wipe_dataset_caches(r, ds_key)


def test_step_reset_repeats():
    """After a forward walk to absorption, reset() returns to the Init snapshot
    and a second forward walk reaches the same absorbing label."""
    rf, irf, def3, ds, ctx = _build()
    icf, x, c_star, v, ds_key, r = _find_continue_icf(irf, ctx, def3, ds, sample_idx=0)

    def _walk_to_absorption(v):
        while v.absorbing is None:
            h_tree, h_leaf = v.rho[0]
            sigma = _leaf_value_on_sample(ctx, irf, x, h_tree, h_leaf)
            v.step(sigma)
        return v.absorbing

    a1 = _walk_to_absorption(v)
    v.reset()
    assert v.absorbing is None
    assert v.rho, "rho should be repopulated after reset"
    a2 = _walk_to_absorption(v)
    assert a1 == a2 == 1
    print(f"[{DATASET}] reset → second forward walk converges to "
          f"absorbing={a2} OK")

    wipe_dataset_caches(r, ds_key)


if __name__ == "__main__":
    test_init_reaches_continue_with_widened_icf()
    test_step_forward_walk_to_absorption()
    test_step_reset_repeats()
    print("OK")
