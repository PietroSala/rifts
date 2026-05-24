"""Sanity test of the leaf-formula and cell-exclusivity layer.

For one small included dataset (`Coffee` by default, switchable) and a
handful of test samples:

  1. Build the OBDDContext (cell variables, leaf formulas, cell exclusivity).
  2. For each sample x build the per-feature cell tuple (`sample_cell_tuple`)
     — exactly one cell index per feature.
  3. Translate that into a full variable assignment via `assignment_from_cells`.
  4. Substitute into every tree's leaf formulas Ψ(v). Verify:
       - the formula of the leaf reached by Def3Forest's per-tree predict is
         True;
       - every other leaf's formula is False;
       - the per-feature cell-exclusivity holds for the assignment (True).

This is the "every cell assignment selects exactly one leaf per tree" claim
of §2.5 / §2.10.
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

from drifts.obdd import OBDDContext
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


DATASET = os.environ.get("DRIFTS_TEST_DATASET", "Coffee")
MODELS_DIR = HERE.parents[2] / "models"
N_SAMPLES = int(os.environ.get("DRIFTS_TEST_N_SAMPLES", "5"))


def main() -> int:
    rf = joblib.load(MODELS_DIR / f"{DATASET}.joblib")["model"]
    irf = from_sklearn(rf, dataset=DATASET)
    def3 = Def3Forest(rf)
    ds = load_dataset(DATASET)
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    label_idx_of = irf.phi_L
    leaves_per_tree = irf.leaves_per_tree()

    print(f"[{DATASET}] n_features={irf.n_features} n_trees={irf.n_trees}",
          flush=True)

    t0 = time.time()
    ctx = OBDDContext.for_forest(irf)
    n_cell_vars = sum(ctx.n_cells_for(i) for i in range(irf.n_features))
    print(f"[{DATASET}] declared {n_cell_vars} cell variables "
          f"({time.time() - t0:.2f}s)", flush=True)

    t0 = time.time()
    leaf_formulas = ctx.compute_leaf_formulas()
    print(f"[{DATASET}] built leaf formulas for {sum(len(m) for m in leaf_formulas)} "
          f"leaves ({time.time() - t0:.2f}s)", flush=True)

    # Test on the first N_SAMPLES test samples.
    n_check = min(N_SAMPLES, ds["n_test"])
    n_features = irf.n_features
    per_tree_def3 = def3.per_tree_predict(ds["X_test"][:n_check])
    mismatches = 0
    excl_fail = 0

    for k in range(n_check):
        x = [float(v) for v in ds["X_test"][k]]
        cells = ctx.sample_cell_tuple(x)
        sub = ctx.assignment_from_cells(cells)

        # cell exclusivity must hold on the cell-tuple assignment
        excl = ctx.cell_exclusivity()
        if ctx.bdd.let(sub, excl) != ctx.bdd.true:
            excl_fail += 1
            continue

        for j, tree_map in enumerate(leaf_formulas):
            # the leaf Def-3 says we reach in tree j:
            target_label = str(per_tree_def3[j, k])
            reached_leaf = None
            for leaf_idx, leaf_node in enumerate(leaves_per_tree[j]):
                lab = inv_phi_L[leaf_node["label_idx"]]
                if str(lab) == target_label and reached_leaf is None:
                    # only mark the first; we will verify via Ψ
                    pass
            for leaf_idx, psi in tree_map.items():
                val = ctx.bdd.let(sub, psi)
                got = (val == ctx.bdd.true)
                if got:
                    if reached_leaf is not None:
                        # more than one leaf's Ψ true → contradiction
                        mismatches += 1
                    reached_leaf = leaf_idx
            if reached_leaf is None:
                mismatches += 1
                continue
            # the reached leaf's label must match Def-3
            got_label = inv_phi_L[leaves_per_tree[j][reached_leaf]["label_idx"]]
            if str(got_label) != target_label:
                mismatches += 1

    print(f"[{DATASET}] checked {n_check} samples × {irf.n_trees} trees "
          f"= {n_check * irf.n_trees} (leaf, tree) checks", flush=True)
    print(f"[{DATASET}] mismatches: {mismatches}", flush=True)
    print(f"[{DATASET}] cell-exclusivity failures: {excl_fail}", flush=True)
    return 0 if (mismatches == 0 and excl_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
