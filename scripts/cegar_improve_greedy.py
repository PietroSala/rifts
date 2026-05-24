#!/usr/bin/env python
"""Check whether existing greedy maximal reasons are improvable under
strict (cell-consistent) CEGAR.

For each (dataset, sample) with a greedy reason in
``code/sweeps/maximal_reasons/sweep.db``, this script runs a one-pass
greedy widener that uses ``CEGARMajority.is_good_strict`` instead of the
conservative ``majority_check``. The strict check accepts widenings
that the conservative Adv_max rejects when the adversarial leaf-tuple
is not cell-realisable.

Filtered to the Max-iAXp-ok overlap by default so we have a meaningful
comparison set (≤ 1240 samples).

Output:
  * stdout per sample: ρ before / after / Δρ / # cache hits learned.
  * CSV at code/sweeps/cegar_improvable/<dataset>__N{n}.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

from drifts.cegar_majority import CEGARMajority
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


MODELS = REPO / "models"
GREEDY_DB = REPO / "sweeps" / "maximal_reasons" / "sweep.db"
MAXIAXP = REPO / "max-iaxp"
OUT_DIR = REPO / "sweeps" / "cegar_improvable"
OUT_DB = OUT_DIR / "sweep.db"


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS cegar_widenings (
    dataset                 TEXT NOT NULL,
    sample                  INTEGER NOT NULL,
    c_star                  INTEGER,
    rho_greedy              INTEGER,
    rho_strict              INTEGER,
    delta_rho               INTEGER,
    n_widens                INTEGER,
    n_passes                INTEGER,
    n_cegar_calls           INTEGER,
    n_no_goods_learned      INTEGER,
    elapsed_s               REAL,
    reason_pos_json         TEXT,
    inserted_at             TEXT,
    PRIMARY KEY (dataset, sample)
);
"""


def _open_db():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUT_DB)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _processed_keys(conn, ds):
    return {row[0] for row in conn.execute(
        "SELECT sample FROM cegar_widenings WHERE dataset=?", (ds,))}


def _maxiaxp_ok():
    out = {}
    for p in sorted(MAXIAXP.glob("*/results.csv")):
        ds = p.parent.name
        df = pd.read_csv(p)
        ok = df[df["solver_status"] == "ok"]
        if ok.empty:
            continue
        out[ds] = ok["sample_idx"].astype(int).tolist()
    return out


def _greedy_reason(g, ds, k):
    row = g.execute(
        "SELECT c_star_idx, reason_pos_json FROM reasons "
        "WHERE dataset=? AND sample=?", (ds, k),
    ).fetchone()
    if not row:
        return None, None
    icf = {int(f): tuple(v) for f, v in json.loads(row[1]).items()}
    return int(row[0]), icf


def _rho_cells(icf):
    return sum(e - b for f, (b, e) in icf.items())


def widen_strict(cegar, irf, eu_sizes, icf, c_star, time_budget_s=None):
    """One-pass-to-fixpoint greedy widener using CEGAR strict check.

    Returns (final_icf, n_widens, n_passes, elapsed_s, n_cegar_calls,
              n_no_goods_learned).
    """
    icf = dict(icf)
    n_widens = 0
    n_passes = 0
    n_cegar = 0
    n_learned_start = cegar.total_no_goods_learned
    t0 = time.time()
    while True:
        n_passes += 1
        improved = False
        for f in range(irf.n_features):
            if time_budget_s and (time.time() - t0) > time_budget_s:
                return (icf, n_widens, n_passes, time.time() - t0,
                        n_cegar, cegar.total_no_goods_learned - n_learned_start)
            n = eu_sizes[f]
            # left
            b, e = icf[f]
            if b > -1:
                cand = dict(icf); cand[f] = (b - 1, e)
                n_cegar += 1
                if cegar.is_good_strict(cand, c_star).verdict == "good":
                    icf = cand; n_widens += 1; improved = True
            # right
            b, e = icf[f]
            if e < n:
                cand = dict(icf); cand[f] = (b, e + 1)
                n_cegar += 1
                if cegar.is_good_strict(cand, c_star).verdict == "good":
                    icf = cand; n_widens += 1; improved = True
        if not improved:
            break
    return (icf, n_widens, n_passes, time.time() - t0,
            n_cegar, cegar.total_no_goods_learned - n_learned_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    help="restrict to these dataset names (default: all "
                         "max-iaxp-ok datasets)")
    ap.add_argument("--n-samples", type=int, default=0,
                    help="samples per dataset (0 = all max-iaxp-ok)")
    ap.add_argument("--time-budget-s", type=float, default=60.0,
                    help="per-sample wall budget for the widener")
    args = ap.parse_args()

    conn = _open_db()
    g = sqlite3.connect(f"file:{GREEDY_DB}?mode=ro", uri=True)
    mxok = _maxiaxp_ok()
    target_ds = args.datasets or list(mxok.keys())
    print(f"datasets: {target_ds}  n_samples each: "
          f"{'ALL' if args.n_samples == 0 else args.n_samples}  "
          f"time_budget_s/sample: {args.time_budget_s}", flush=True)
    print(f"output DB: {OUT_DB}", flush=True)

    overall_summary = []
    for ds in target_ds:
        if ds not in mxok:
            print(f"  {ds}: no max-iaxp-ok rows — skipping")
            continue
        ks_all = mxok[ds]
        if args.n_samples > 0:
            ks_all = ks_all[:args.n_samples]
        done = _processed_keys(conn, ds)
        ks = [k for k in ks_all if k not in done]
        print(f"\n=== {ds}: {len(ks)} samples to process "
              f"(skipping {len(ks_all) - len(ks)} already done) ===",
              flush=True)
        if not ks:
            continue

        rf = joblib.load(MODELS / f"{ds}.joblib")["model"]
        irf = from_sklearn(rf, dataset=ds)
        ctx = OBDDContext.for_forest(irf, encoding="binary")
        ctx.bootstrap(BootstrapConfig(mode="per_worker"))
        eu_sizes = [len(irf.EU[i]) for i in range(irf.n_features)]
        cegar = CEGARMajority(irf=irf, ctx=ctx)

        n_improved = 0
        sum_dr = 0
        sum_t = 0.0
        n_done = 0
        for k in ks:
            c_star, greedy_icf = _greedy_reason(g, ds, k)
            if greedy_icf is None:
                print(f"  k={k}: no greedy row — skipping", flush=True)
                continue
            rho0 = _rho_cells(greedy_icf)
            new_icf, nw, npass, t, ncegar, learned = widen_strict(
                cegar, irf, eu_sizes, greedy_icf, c_star,
                time_budget_s=args.time_budget_s,
            )
            rho1 = _rho_cells(new_icf)
            dr = rho1 - rho0
            n_improved += int(dr > 0)
            sum_dr += dr
            sum_t += t
            n_done += 1
            conn.execute(
                """
                INSERT OR REPLACE INTO cegar_widenings
                  (dataset, sample, c_star, rho_greedy, rho_strict,
                   delta_rho, n_widens, n_passes, n_cegar_calls,
                   n_no_goods_learned, elapsed_s, reason_pos_json, inserted_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (ds, int(k), int(c_star), rho0, rho1, dr, nw, npass, ncegar,
                 learned, round(t, 4),
                 json.dumps({str(f): list(v) for f, v in new_icf.items()}),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            conn.commit()
            if n_done % 20 == 0 or dr > 0:
                print(f"  k={k:>5d}  c*={c_star}  ρ {rho0:>4d} → {rho1:>4d}  "
                      f"Δρ=+{dr:<3d}  widens={nw:>3d}  calls={ncegar:>5d}  "
                      f"learned={learned:>4d}  t={t:.2f}s",
                      flush=True)

        print(f"\n  {ds}: improved {n_improved}/{n_done}  "
              f"Σ Δρ={sum_dr}  Σ t={sum_t:.1f}s  "
              f"cache: {len(cegar.infeasibility_cache)} no-goods")
        overall_summary.append({
            "dataset": ds, "n": n_done,
            "n_improved": n_improved, "sum_dr": sum_dr,
            "sum_t": round(sum_t, 2),
            "cache": len(cegar.infeasibility_cache),
        })

    print("\n=== OVERALL ===")
    for r in overall_summary:
        print(f"  {r['dataset']:<22s}  n={r['n']:>4d}  "
              f"improved={r['n_improved']:>3d}  Σ Δρ={r['sum_dr']:>5d}  "
              f"Σ t={r['sum_t']:>8.1f}s  cache={r['cache']}")


if __name__ == "__main__":
    main()
