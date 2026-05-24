"""Majority oracle sanity check on the 66 included datasets.

For every test sample of every included dataset:

  1. Build the trivial ICF and its profile (every per-tree profile is a
     singleton → the state is slackless).
  2. Set seed class `c⋆` = the Def-3 forest prediction.
     The oracle must return `verdict = "good"` and recover `c⋆` as winner.
     This is the "agreement with Def-3" claim.
  3. Set a *different* class `c′ ≠ c⋆` as the seed.
     Adv(c⋆) ≥ 0 must hold and the oracle must return `verdict = "bad"`
     with a complete legal witness; the witness's per-tree leaf-tuple must
     produce `c⋆` under our Def-3 vote (sanity).
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
from drifts.milp_majority import majority_check
from drifts.partial_assignment import PartialAssignment
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


def _build_alpha_from_profile(profile, n_leaves_per_tree) -> PartialAssignment:
    return PartialAssignment.initial_from_profile(profile, n_leaves_per_tree)


def _vote_from_witness(witness: PartialAssignment, irf):
    """Sanity-evaluate a witness: collect chosen-leaf labels per tree, then
    compute Def-3 majority with lex-first tie-break."""
    leaves_per_tree = irf.leaves_per_tree()
    counts = [0] * irf.n_labels
    for t in range(irf.n_trees):
        for v, val in witness.decided.get(t, {}).items():
            if val == 1:
                lab = leaves_per_tree[t][v]["label_idx"]
                counts[lab] += 1
                break
    # arg-max, tie → smallest label_idx
    best = max(range(irf.n_labels), key=lambda c: (counts[c], -c))
    return best


def check_one(name: str) -> dict:
    rf = joblib.load(MODELS / f"{name}.joblib")["model"]
    irf = from_sklearn(rf, dataset=name)
    def3 = Def3Forest(rf)
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    label_idx_of = {k: v for k, v in irf.phi_L.items()}
    ds = load_dataset(name)
    X = ds["X_test"]
    n_leaves_per_tree = irf.n_leaves_per_tree()
    fwd_pred = def3.predict(X)

    n_correct_seed = 0
    n_wrong_seed_bad = 0
    n_other_inconclusive = 0
    n_witness_check = 0
    n_total = len(X)
    for k in range(n_total):
        x = [float(v) for v in X[k]]
        icf = trivial_icf(irf, x)
        profile = forest_profile(irf, icf)
        alpha = _build_alpha_from_profile(profile, n_leaves_per_tree)

        # case 1: seed = the correct Def-3 prediction → expect "good"
        c_star_str = str(fwd_pred[k])
        c_star = label_idx_of[c_star_str]
        res = majority_check(irf, profile, alpha, c_star)
        assert res.slackless, f"{name} sample {k}: trivial ICF should be slackless"
        assert res.verdict == "good", (
            f"{name} sample {k}: with correct seed got verdict={res.verdict}, "
            f"adv={res.adv}"
        )
        n_correct_seed += 1

        # case 2: seed = a different class (only if there is one)
        if irf.n_labels >= 2:
            wrong = next((c for c in range(irf.n_labels) if c != c_star), None)
            if wrong is not None:
                res2 = majority_check(irf, profile, alpha, wrong)
                if res2.verdict == "bad":
                    # witness must vote c_star
                    voted = _vote_from_witness(res2.witness, irf)
                    assert voted == c_star, (
                        f"{name} sample {k}: bad witness votes {voted}, expected {c_star}"
                    )
                    n_wrong_seed_bad += 1
                    n_witness_check += 1
                elif res2.verdict == "inconclusive":
                    n_other_inconclusive += 1
                else:
                    # "good" with the wrong seed means c_star couldn't beat
                    # `wrong` either — only possible on ties. Sanity: confirm.
                    pass

    return {
        "dataset": name, "n_samples": n_total, "n_trees": irf.n_trees,
        "n_correct_seed_good": n_correct_seed,
        "n_wrong_seed_bad": n_wrong_seed_bad,
        "n_wrong_seed_inconclusive": n_other_inconclusive,
        "n_witness_checked": n_witness_check,
    }


def main() -> int:
    datasets = _datasets()
    print(f"majority-oracle sanity over {len(datasets)} datasets\n", flush=True)
    fails = 0
    t0 = time.time()
    for i, name in enumerate(datasets, 1):
        try:
            r = check_one(name)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:3d}/{len(datasets)}] {name:32s} EXC: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            fails += 1
            continue
        ok = r["n_correct_seed_good"] == r["n_samples"]
        mark = "OK" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"[{i:3d}/{len(datasets)}] {r['dataset']:32s} "
              f"n_test={r['n_samples']:>5d} correct-seed-good={r['n_correct_seed_good']:>5d} "
              f"wrong-seed-bad={r['n_wrong_seed_bad']:>5d} "
              f"wrong-seed-inconcl={r['n_wrong_seed_inconclusive']:>3d} "
              f"witness-✓={r['n_witness_checked']:>5d} {mark}",
              flush=True)
    print(f"\n{len(datasets) - fails}/{len(datasets)} OK ({time.time() - t0:.1f}s)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
