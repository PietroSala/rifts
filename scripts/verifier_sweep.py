#!/usr/bin/env python
"""Verifier sweep across the 66 included datasets — with shared caches.

Caches are scoped per ``(dataset, c⋆)`` and persist ACROSS samples in the
same dataset/c⋆ pair (so sample k+1 may hit a G/B/C entry inserted by
sample k). F and C, being dataset-wide, also accumulate across samples
of every c⋆ in the dataset.

Per dataset, three scenarios are run on a small sample budget:

  * trivial-ICF, correct seed     → expect absorbing = 1
  * trivial-ICF, wrong seed       → expect BadException
  * widened-ICF, correct seed     → expect Init → Continue, the real-sample
                                    forward walk reaches absorbing = 1,
                                    reset() reproduces

Outputs a CSV at ``code/sweeps/verifier_sweep.csv`` and a per-sample CSV
at ``code/sweeps/verifier_sweep_samples.csv`` with cache-hit telemetry
per sample (so cross-sample sharing is observable).
"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
import traceback
from pathlib import Path

import joblib

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

from cache.caches import open_caches
from cache.connection import get_client
from drifts.icf import trivial_icf
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.profile import forest_profile
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn
from drifts.verifier import EPSILON, BadException, Verifier

from load_ucr import load_dataset  # noqa: E402


MODELS = REPO / "models"
ORDER = REPO / "experiments_order" / "included_topo.csv"
OUT = REPO / "sweeps" / "verifier_sweep.csv"
OUT_SAMPLES = REPO / "sweeps" / "verifier_sweep_samples.csv"

N_SAMPLES = int(os.environ.get("VERIFIER_SWEEP_N_SAMPLES", "3"))
ENCODING = os.environ.get("VERIFIER_SWEEP_ENCODING", "binary")
SESSION = int(time.time() * 1000) ^ os.getpid()


def _datasets():
    with ORDER.open() as f:
        return [r["dataset"] for r in csv.DictReader(f) if r.get("dataset")]


def _widen(icf, irf, delta: int):
    out = {}
    for i, (b, e) in icf.items():
        n = len(irf.EU[i])
        out[i] = (max(-1, b - delta), min(n, e + delta))
    return out


def _leaf_value(ctx, x, t, v):
    cells = ctx.sample_cell_tuple(x)
    sub = ctx.assignment_from_cells(cells)
    psi = ctx.leaf_formulas[t][v]
    return 1 if ctx.bdd.let(sub, psi) == ctx.bdd.true else 0


def _forward_walk(v, ctx, x, max_steps_factor: int = 10) -> int:
    cap = max_steps_factor * v.irf.n_trees * max(v.irf.n_leaves_per_tree())
    n = 0
    while v.absorbing is None:
        if not v.rho:
            raise RuntimeError("rho empty without absorbing")
        h_tree, h_leaf = v.rho[0]
        sigma = _leaf_value(ctx, x, h_tree, h_leaf)
        v.step(sigma)
        n += 1
        if n > cap:
            raise RuntimeError(f"forward walk too long ({n} > {cap})")
    return n


def _try_continue(r, ds_key: str, irf, ctx, c_star: int, icf, k: int):
    """Open a (shared) cache for (ds_key, c_star) and run Init on the given
    ICF. Returns (verifier, reached) where reached ∈ {"continue", "absorbed",
    "bad"}. The caller is responsible for caching the (dataset, c_star) →
    cache binding if it wants reuse across calls.
    """
    caches = open_caches(r, ds_key, c_star)
    v = Verifier(irf, ctx, caches, c_star, rng=random.Random(k))
    try:
        outcome = v.init(icf)
    except BadException:
        return v, "bad"
    if outcome is None and v.absorbing is None and v.rho:
        return v, "continue"
    return v, "absorbed"


def sweep_one(name: str):
    t0 = time.time()
    rf = joblib.load(MODELS / f"{name}.joblib")["model"]
    irf = from_sklearn(rf, dataset=name)
    def3 = Def3Forest(rf)
    ds = load_dataset(name)
    ctx = OBDDContext.for_forest(irf, encoding=ENCODING)
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    t_build = time.time() - t0

    fwd = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}
    n_check = min(N_SAMPLES, ds["n_test"])

    # Single dataset prefix → shared (dataset, c_star) caches across samples.
    ds_key = f"_vsweep_{name}_{SESSION}"
    r = get_client("DATA")

    n_correct = 0
    n_wrong_bad = 0
    n_continue_reached = 0
    n_continue_walk_good = 0
    n_reset_reproduce = 0
    failures = []
    per_sample = []                # row dicts for OUT_SAMPLES

    t_run0 = time.time()
    for k in range(n_check):
        x = [float(v) for v in ds["X_test"][k]]
        icf = trivial_icf(irf, x)
        c_star = label_idx_of[str(fwd[k])]

        # ------------------------------------------------------------- correct
        caches = open_caches(r, ds_key, c_star)
        # snapshot cache sizes BEFORE refresh — that is what the verifier sees
        # locally; after `init()` refresh_all populates `local_entries` from
        # peers (here: prior samples sharing this (ds_key, c_star)).
        before = {"G": len(caches.G), "B": len(caches.B),
                  "F": len(caches.F), "C": len(caches.C)}
        v1 = Verifier(irf, ctx, caches, c_star)
        try:
            outcome = v1.init(icf)
            corr_ok = (outcome == 1)
            corr_outcome = outcome
        except BadException as e:
            corr_ok = False
            corr_outcome = f"Bad({e.reason})"
            failures.append(f"sample {k}: correct seed → unexpected {corr_outcome}")
        after = {"G": len(caches.G), "B": len(caches.B),
                 "F": len(caches.F), "C": len(caches.C)}
        if corr_ok:
            n_correct += 1

        # ------------------------------------------------------------- wrong
        wrong_ok = None
        if irf.n_labels >= 2:
            wrong = (c_star + 1) % irf.n_labels   # consistent per dataset (modulo c_star)
            caches_w = open_caches(r, ds_key, wrong)
            v2 = Verifier(irf, ctx, caches_w, wrong)
            try:
                v2.init(icf)
                wrong_ok = False
                failures.append(f"sample {k}: wrong seed did not raise Bad")
            except BadException:
                wrong_ok = True
                n_wrong_bad += 1

        # ------------------------------------------------------------- widened
        widen_reached = "skipped"
        widen_walk = None
        widen_reset = None
        trivial = trivial_icf(irf, x)
        for delta in (1, 2, 3, 4):
            wicf = _widen(trivial, irf, delta)
            prof = forest_profile(irf, wicf)
            if all(len(p) == 1 for p in prof):
                continue
            v3, reached = _try_continue(r, ds_key, irf, ctx, c_star, wicf, k)
            widen_reached = reached
            if reached == "continue":
                n_continue_reached += 1
                try:
                    _forward_walk(v3, ctx, x)
                    widen_walk = v3.absorbing
                    if v3.absorbing == 1:
                        n_continue_walk_good += 1
                    else:
                        failures.append(
                            f"sample {k}: widened walk absorbed to "
                            f"{v3.absorbing}"
                        )
                    v3.reset()
                    _forward_walk(v3, ctx, x)
                    widen_reset = v3.absorbing
                    if v3.absorbing == 1:
                        n_reset_reproduce += 1
                except Exception as e:
                    failures.append(f"sample {k}: widened walk failed — {e}")
            break   # first widening that didn't absorb at Init is our test

        per_sample.append({
            "dataset": name,
            "sample": k,
            "c_star": c_star,
            "correct_outcome": corr_outcome,
            "wrong_raised_bad": wrong_ok,
            "widen_reached": widen_reached,
            "widen_walk_absorbing": widen_walk,
            "widen_reset_absorbing": widen_reset,
            "G_before": before["G"], "G_after": after["G"],
            "B_before": before["B"], "B_after": after["B"],
            "F_before": before["F"], "F_after": after["F"],
            "C_before": before["C"], "C_after": after["C"],
        })

    t_run = time.time() - t_run0
    # Final cache sizes for telemetry
    caches_final = open_caches(r, ds_key, 0)
    caches_final.refresh_all()
    return {
        "dataset": name,
        "n_trees": irf.n_trees,
        "n_features": irf.n_features,
        "n_labels": irf.n_labels,
        "n_samples_checked": n_check,
        "t_build_s": round(t_build, 2),
        "t_run_s": round(t_run, 2),
        "correct_good": n_correct,
        "wrong_bad": n_wrong_bad,
        "continue_reached": n_continue_reached,
        "continue_walk_good": n_continue_walk_good,
        "reset_reproduce": n_reset_reproduce,
        "F_final": len(caches_final.F),
        "C_final": len(caches_final.C),
        "n_failures": len(failures),
        "first_failure": failures[0] if failures else "",
    }, per_sample


def main():
    OUT.parent.mkdir(exist_ok=True, parents=True)
    datasets = _datasets()
    print(f"Verifier sweep across {len(datasets)} datasets, "
          f"N_SAMPLES={N_SAMPLES}, encoding={ENCODING}, "
          f"session={SESSION}", flush=True)
    summary_rows = []
    sample_rows = []
    t_total0 = time.time()
    for i, name in enumerate(datasets, start=1):
        try:
            row, per_sample = sweep_one(name)
        except Exception as exc:
            row = {
                "dataset": name,
                "n_trees": -1, "n_features": -1, "n_labels": -1,
                "n_samples_checked": 0,
                "t_build_s": 0.0, "t_run_s": 0.0,
                "correct_good": 0, "wrong_bad": 0,
                "continue_reached": 0, "continue_walk_good": 0,
                "reset_reproduce": 0,
                "F_final": -1, "C_final": -1,
                "n_failures": 1, "first_failure": f"crash: {exc!r}",
            }
            per_sample = []
            traceback.print_exc()
        summary_rows.append(row)
        sample_rows.extend(per_sample)
        print(
            f"[{i:>2d}/{len(datasets)}] {name:<32s} "
            f"correct={row['correct_good']}/{row['n_samples_checked']} "
            f"wrong_bad={row['wrong_bad']}/{row['n_samples_checked']} "
            f"continue={row['continue_walk_good']}/{row['continue_reached']} "
            f"F={row['F_final']} C={row['C_final']} "
            f"fail={row['n_failures']} "
            f"t={row['t_build_s']+row['t_run_s']:.1f}s",
            flush=True,
        )

    with OUT.open("w", newline="") as f:
        cols = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)
    if sample_rows:
        with OUT_SAMPLES.open("w", newline="") as f:
            cols = list(sample_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(sample_rows)

    elapsed = time.time() - t_total0
    n_total = sum(r["n_samples_checked"] for r in summary_rows)
    n_correct = sum(r["correct_good"] for r in summary_rows)
    n_wrong_bad = sum(r["wrong_bad"] for r in summary_rows)
    n_continue = sum(r["continue_reached"] for r in summary_rows)
    n_continue_ok = sum(r["continue_walk_good"] for r in summary_rows)
    n_reset_ok = sum(r["reset_reproduce"] for r in summary_rows)
    n_fail = sum(r["n_failures"] for r in summary_rows)
    print(f"\n  Summary CSV  → {OUT}")
    print(f"  Per-sample   → {OUT_SAMPLES}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  Trivial correct seed Good:  {n_correct} / {n_total}")
    print(f"  Trivial wrong   seed Bad:   {n_wrong_bad} / {n_total}")
    print(f"  Widened Continue reached:   {n_continue} / {n_total}")
    print(f"  Widened walk → absorbing=1: {n_continue_ok} / {n_continue}")
    print(f"  Reset reproduces walk:      {n_reset_ok} / {n_continue}")
    print(f"  Total failures (warnings):  {n_fail}")


if __name__ == "__main__":
    main()
