#!/usr/bin/env python
"""Continuous refinement sweeper with per-sample doubling budget.

Loop, forever:

  1. Pick the next sample to refine. Priority:
       a) the oldest **greedy-only** sample (no refinement chain yet);
       b) else the oldest **refined-but-uncertified** sample (chain in
          the refinement DB whose ``certified_maximum=0``).
  2. Compute its budget: ``60 × 2^n_prior_attempts`` seconds.
  3. Run ``refine_sample`` with that wall-clock cap, CEGAR-strict
     verdict, significant-test branching.
  4. Loop.

Scope:

  ``--scope general``  every (dataset, sample) the greedy sweeper has
                        produced.
  ``--scope axp``      restricted to the 1240 Max-iAXp-overlap samples.

The greedy sweeper writes new ``reasons`` rows live, so this loop
naturally picks them up as they arrive. If the greedy is slow, this
loop falls back to re-refining the oldest uncertified sample with its
(now-doubled) budget — exactly per the design.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
import traceback
from pathlib import Path

import joblib
import pandas as pd

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

import refinement_sweep as RS
from refinement_sweep import (
    _open_db, refine_sample, _rho,
    GREEDY_DB,
)

from drifts.cegar_majority import CEGARMajority
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


MAXIAXP = REPO / "max-iaxp"
MODELS = REPO / "models"

# Force CEGAR + significant-test branching in the refinement.
RS.USE_CEGAR = True
RS.USE_SIGNIFICANT = True


def _maxiaxp_ok():
    out = set()
    for p in sorted(MAXIAXP.glob("*/results.csv")):
        ds = p.parent.name
        df = pd.read_csv(p)
        for k in df.loc[df["solver_status"] == "ok", "sample_idx"].astype(int):
            out.add((ds, int(k)))
    return out


def _pick_next_sample(conn_g, conn_r, scope_filter):
    """Return (dataset, sample) of the next sample to refine, or None.

    Priority a): oldest greedy-only row whose (ds, sample) is NOT in
    ``refinement_chain_summary``.
    Priority b): oldest ``refinement_chain_summary`` row with
    ``certified_maximum=0``, ordered by ``inserted_at`` ascending.
    """
    # Pull all greedy keys (small enough at any point).
    greedy_keys = [(ds, k) for (ds, k) in conn_g.execute(
        "SELECT dataset, sample FROM reasons ORDER BY inserted_at ASC"
    )]
    if scope_filter is not None:
        greedy_keys = [k for k in greedy_keys if k in scope_filter]
    if not greedy_keys:
        return None

    chained = set()
    for ds, k in conn_r.execute(
        "SELECT dataset, sample FROM refinement_chain_summary"
    ):
        chained.add((ds, k))

    # (a) oldest greedy-only sample.
    for k in greedy_keys:
        if k not in chained:
            return k

    # (b) oldest refined-not-certified.
    row = conn_r.execute(
        "SELECT dataset, sample FROM refinement_chain_summary "
        "WHERE certified_maximum = 0 ORDER BY inserted_at ASC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    if scope_filter is not None and (row[0], int(row[1])) not in scope_filter:
        # The oldest uncertified isn't in our scope; scan for one that is.
        for ds, k in conn_r.execute(
            "SELECT dataset, sample FROM refinement_chain_summary "
            "WHERE certified_maximum = 0 ORDER BY inserted_at ASC"
        ):
            if (ds, int(k)) in scope_filter:
                return (ds, int(k))
        return None
    return (row[0], int(row[1]))


def _greedy_reason(conn_g, ds, k):
    row = conn_g.execute(
        "SELECT c_star_idx, reason_pos_json FROM reasons "
        "WHERE dataset=? AND sample=?", (ds, k),
    ).fetchone()
    if row is None:
        return None
    raw = json.loads(row[1])
    return int(row[0]), {int(f): tuple(v) for f, v in raw.items()}


def _n_prior_attempts(conn_r, ds, k):
    row = conn_r.execute(
        "SELECT COUNT(*) FROM refinements WHERE dataset=? AND sample=?",
        (ds, k),
    ).fetchone()
    return int(row[0]) if row else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("general", "axp"), required=True)
    ap.add_argument("--base-cap-s", type=float, default=60.0,
                    help="initial wall budget per sample (seconds)")
    ap.add_argument("--poll-s", type=float, default=5.0,
                    help="sleep between empty-queue polls")
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args()

    scope = _maxiaxp_ok() if args.scope == "axp" else None
    if scope is not None:
        print(f"AXp scope: {len(scope)} sample keys", flush=True)
    conn_g = sqlite3.connect(f"file:{GREEDY_DB}?mode=ro", uri=True)
    conn_r = _open_db()

    # Per-dataset CEGAR/OBDD cache so we don't re-bootstrap.
    cache = {}
    def _ensure(ds):
        if ds in cache:
            return cache[ds]
        rf = joblib.load(MODELS / f"{ds}.joblib")["model"]
        irf = from_sklearn(rf, dataset=ds)
        xs = load_dataset(ds)["X_test"]
        ctx = OBDDContext.for_forest(irf, encoding="binary")
        ctx.bootstrap(BootstrapConfig(mode="per_worker"))
        cegar = CEGARMajority(irf=irf, ctx=ctx)
        cache[ds] = (irf, xs, cegar)
        return cache[ds]

    n_processed = 0
    n_certified = 0
    empty_polls = 0
    t_loop0 = time.time()
    while True:
        pick = _pick_next_sample(conn_g, conn_r, scope)
        if pick is None:
            empty_polls += 1
            time.sleep(args.poll_s)
            if empty_polls % 12 == 0:
                # Re-check after one minute total
                print(f"  ({args.scope}) idle, no sample to refine; "
                      f"sleeping {args.poll_s}s…", flush=True)
            continue
        empty_polls = 0
        ds, k = pick

        n_prior = _n_prior_attempts(conn_r, ds, k)
        budget = args.base_cap_s * (2 ** n_prior)

        gr = _greedy_reason(conn_g, ds, k)
        if gr is None:
            time.sleep(0.1)
            continue
        c_star, greedy_icf = gr
        irf, xs, cegar = _ensure(ds)
        x = [float(v) for v in xs[k]]

        t0 = time.time()
        res = refine_sample(
            conn_r, irf, c_star, x, greedy_icf, ds, int(k),
            random.Random(args.rng_seed + 10_000 * int(k) + hash(ds) % 10_000),
            max_nodes=None, max_time_s=budget,
            cegar=cegar,
        )
        t = time.time() - t0
        if res is None:
            # Should not happen given pick logic, but defensive.
            continue
        r0, r1, n_ref, n_imp, t_total, closure = res
        n_processed += 1
        cert = (closure == "exhausted")
        if cert:
            n_certified += 1
        tag = "CERT" if cert else closure
        print(f"[{args.scope} {n_processed}] {ds:<22s} k={k:>5d}  "
              f"attempts so far={n_prior + 1}  budget={budget:.0f}s  "
              f"ρ {r0}→{r1}  +{r1-r0}  refs={n_ref} imp={n_imp} "
              f"[{tag}] t={t:.2f}s  cegar_cache={len(cegar.infeasibility_cache)}",
              flush=True)


if __name__ == "__main__":
    main()
