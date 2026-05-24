#!/usr/bin/env python
"""Progressive-time-cap refinement over the Max-iAXp-ok samples.

For each (dataset, sample_idx) where Max-iAXp reported ``solver_status=ok``
in ``results/max-iaxp/<dataset>/results.csv``, run the refinement chain
with a doubling time cap (1 s → 2 s → 4 s → ...) until either:

  * the chain becomes ``exhausted`` (certified maximum — greedy was or is
    the maximum), or
  * the maximum time cap is reached (default 1024 s ≈ 17 min, matching
    the Max-iAXp ladder's outer bound).

Each tier's call uses the existing resume logic: cap_hit_time leaves
from tier N are picked up by tier N+1, so we don't lose work between
budgets.

Output goes to ``code/sweeps/refinements/sweep.db`` — the same DB used
by the unconstrained refinement sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

# Import the existing primitives — refinement_sweep does all the heavy lifting.
import refinement_sweep as RS
from refinement_sweep import (
    _open_db, refine_sample, _resume_state, _rho,
    MODELS, GREEDY_DB,
)

from load_ucr import load_dataset  # noqa: E402
from drifts.sklearn_io import from_sklearn
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.cegar_majority import CEGARMajority

USE_CEGAR = os.environ.get("REFINEMENT_USE_CEGAR", "0") in ("1", "true", "True")

MAXIAXP_DIR = REPO / "max-iaxp"
# Default ladder (seconds) when no --max-time-s override is given.
BUDGETS = [60.0, 120.0, 240.0, 480.0, 960.0]


def _maxiaxp_ok_samples():
    """Return ``[(dataset, sample_idx), ...]`` for every Max-iAXp ok row."""
    out = []
    for p in sorted(MAXIAXP_DIR.glob("*/results.csv")):
        ds = p.parent.name
        df = pd.read_csv(p)
        ok = df[df["solver_status"] == "ok"]
        for k in ok["sample_idx"].astype(int).unique():
            out.append((ds, int(k)))
    return out


def _greedy_reason(conn_g, ds, k, conn_cegar=None):
    """Best available reason as starting point.

    If ``conn_cegar`` is provided and a CEGAR-widened reason exists for
    ``(ds, k)``, use that (strictly ≥ greedy). Otherwise fall back to the
    plain greedy reason.
    """
    if conn_cegar is not None:
        row = conn_cegar.execute(
            "SELECT c_star, reason_pos_json FROM cegar_widenings "
            "WHERE dataset=? AND sample=?", (ds, k),
        ).fetchone()
        if row is not None:
            raw = json.loads(row[1])
            icf = {int(f): tuple(v) for f, v in raw.items()}
            return int(row[0]), icf
    row = conn_g.execute(
        "SELECT c_star_idx, reason_pos_json FROM reasons "
        "WHERE dataset=? AND sample=?", (ds, k),
    ).fetchone()
    if row is None:
        return None
    raw = json.loads(row[1])
    icf = {int(f): tuple(v) for f, v in raw.items()}
    return int(row[0]), icf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    help="Process only these dataset names "
                         "(default: every max-iAXp-ok dataset)")
    ap.add_argument("--max-time-s", type=float,
                    help="Single per-sample time cap in seconds; "
                         "if set, skips the BUDGETS ladder.")
    args = ap.parse_args()

    conn = _open_db()
    g_conn = sqlite3.connect(f"file:{GREEDY_DB}?mode=ro", uri=True)
    cegar_db_path = REPO / "sweeps" / "cegar_improvable" / "sweep.db"
    cegar_conn = (sqlite3.connect(f"file:{cegar_db_path}?mode=ro", uri=True)
                  if cegar_db_path.exists() else None)
    if USE_CEGAR:
        print(f"CEGAR strict verdict enabled "
              f"({'with' if cegar_conn else 'without'} pre-widened reasons)",
              flush=True)

    todo = _maxiaxp_ok_samples()
    if args.datasets:
        sel = set(args.datasets)
        todo = [t for t in todo if t[0] in sel]
    ladder = [args.max_time_s] if args.max_time_s else BUDGETS
    print(f"Max-iAXp-ok samples: {len(todo)} "
          f"(datasets filter: {sorted(set(t[0] for t in todo))})", flush=True)
    print(f"Budget ladder (seconds): {ladder}", flush=True)
    print(f"Output DB: {RS.OUT_DB}", flush=True)
    print()

    t_total0 = time.time()
    n_certified = 0
    n_no_greedy = 0
    n_processed = 0
    by_ds_certified = {}

    for tier, cap in enumerate(ladder):
        # Skip already-certified samples (built up from the previous tier).
        already = RS._already_certified(conn)
        remaining = [(ds, k) for ds, k in todo if (ds, k) not in already]
        n_certified = len(already & set(todo))
        print(f"=== tier {tier} — MAX_TIME_S = {cap:>5.0f} s — "
              f"remaining = {len(remaining)}/{len(todo)} "
              f"(certified so far = {n_certified}) ===", flush=True)
        if not remaining:
            print("  every Max-iAXp-ok sample is certified; stopping early.")
            break

        # Cache per-dataset state so we don't reload IRF / dataset N times.
        cache = {}
        cegar_by_ds = {}
        for i, (ds, k) in enumerate(remaining, start=1):
            if ds not in cache:
                rf = joblib.load(MODELS / f"{ds}.joblib")["model"]
                irf = from_sklearn(rf, dataset=ds)
                xs = load_dataset(ds)["X_test"]
                cache[ds] = (irf, xs)
                if USE_CEGAR:
                    ctx = OBDDContext.for_forest(irf, encoding="binary")
                    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
                    cegar_by_ds[ds] = CEGARMajority(irf=irf, ctx=ctx)
            irf, xs = cache[ds]
            cegar_instance = cegar_by_ds.get(ds)

            gr = _greedy_reason(g_conn, ds, k, conn_cegar=cegar_conn)
            if gr is None:
                n_no_greedy += 1
                continue
            c_star, greedy_icf = gr
            x = [float(v) for v in xs[k]]

            res = refine_sample(
                conn, irf, c_star, x, greedy_icf, ds, k,
                random.Random(0 + 10_000 * k + hash(ds) % 10_000),
                max_nodes=None, max_time_s=cap,
                cegar=cegar_instance,
            )
            n_processed += 1
            if res is None:
                continue
            r0, r1, n_ref, n_imp, t, closure = res
            tag = "CERT" if closure == "exhausted" else closure
            print(f"  [tier {tier} {i:>4d}/{len(remaining)}] {ds:<28s} "
                  f"k={k:>4d}  ρ {r0:>4d} → {r1:>4d}  +{r1-r0:<3d}  "
                  f"refs={n_ref} imp={n_imp} [{tag}] t={t:.2f}s",
                  flush=True)
            if closure == "exhausted":
                by_ds_certified.setdefault(ds, 0)
                by_ds_certified[ds] += 1

    final_certified = RS._already_certified(conn) & set(todo)
    print()
    print(f"=== FINAL ===")
    print(f"  total Max-iAXp-ok samples:     {len(todo)}")
    print(f"  samples missing in greedy DB:  {n_no_greedy}")
    print(f"  certified (exhausted) after sweep: {len(final_certified)}")
    print(f"  elapsed: {time.time() - t_total0:.1f}s")
    print()
    print(f"Per-dataset certifications produced this run:")
    for ds, n in sorted(by_ds_certified.items()):
        print(f"  {ds:<28s} +{n} certifications")


if __name__ == "__main__":
    main()
