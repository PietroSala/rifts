#!/usr/bin/env python
"""A/B comparison of the two refinement branching strategies.

For a chosen dataset, runs refinement on the same N samples first with
the dichotomic max-free-cells split and then with the significant-test
split. Each chain restarts from the trivial root; the bound starts at
the greedy reason's ρ; the per-attempt cap is identical for both runs.
Reports ρ improvement, closure outcome, attempts / improvements, wall
time, and node count.

The script does not write to the production refinement DB — both runs
are in-memory only via direct calls to ``refine_once``.

Usage:
  python scripts/compare_branching.py --dataset Coffee --n-samples 10 --cap-s 30
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import joblib

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

import refinement_sweep as RS
from drifts.icf import trivial_icf
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


GREEDY_DB = REPO / "sweeps" / "maximal_reasons" / "sweep.db"
MODELS = REPO / "models"


def chain_once(irf, c_star, x, greedy_icf, cap_s, max_nodes, use_sig: bool):
    """One full refinement chain (restart from trivial; same bound across
    attempts) under the chosen branching strategy. Returns telemetry."""
    RS.USE_SIGNIFICANT = use_sig

    n_leaves = irf.n_leaves_per_tree()
    eu_sizes = [len(irf.EU[i]) for i in range(irf.n_features)]
    bound = RS._rho(greedy_icf)
    trivial = trivial_icf(irf, x)
    rng = random.Random(42)

    leaves_state = None
    g_cache: list = []
    b_cache: list = []
    current_rho = bound
    n_attempts = 0
    n_imp = 0
    total_visited = 0
    counters_total = {"bad": 0, "bb_pruned": 0, "saturated": 0,
                      "max_eq": 0, "g_hits": 0, "b_hits": 0}
    t0 = time.time()
    while True:
        (new_icf, found_rho, n_visited, closure,
         counters, leaves_state, g_cache, b_cache) = RS.refine_once(
            irf, c_star, n_leaves, eu_sizes,
            trivial, current_rho, rng,
            max_nodes=max_nodes, max_time_s=cap_s,
            initial_leaves=leaves_state,
            g_cache=g_cache, b_cache=b_cache,
        )
        n_attempts += 1
        total_visited += n_visited
        for k in counters_total:
            counters_total[k] += counters.get(k, 0)
        if closure in ("improvement_found", "improvement_at_max"):
            n_imp += 1
            current_rho = found_rho
            continue
        break
    return {
        "rho_start": bound,
        "rho_end": current_rho,
        "delta_rho": current_rho - bound,
        "closure": closure,
        "n_attempts": n_attempts,
        "n_imp": n_imp,
        "n_visited": total_visited,
        "n_open_remaining": len(leaves_state),
        "elapsed_s": time.time() - t0,
        "counters": counters_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Coffee")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--cap-s", type=float, default=30.0,
                    help="per-attempt time cap (s); applies to both strategies")
    ap.add_argument("--max-nodes", type=int, default=None)
    args = ap.parse_args()

    rf = joblib.load(MODELS / f"{args.dataset}.joblib")["model"]
    irf = from_sklearn(rf, dataset=args.dataset)
    ds = load_dataset(args.dataset)

    g = sqlite3.connect(f"file:{GREEDY_DB}?mode=ro", uri=True)
    rows = g.execute(
        "SELECT sample, c_star_idx, reason_pos_json FROM reasons "
        "WHERE dataset=? ORDER BY sample LIMIT ?",
        (args.dataset, args.n_samples),
    ).fetchall()
    g.close()
    if not rows:
        print(f"No greedy rows for {args.dataset} (greedy v3 hasn't done it yet?)")
        return

    print(f"=== {args.dataset}: {len(rows)} samples, cap={args.cap_s:.0f} s, "
          f"max_nodes={args.max_nodes} ===")
    print()
    hdr = (f"{'k':>4}  {'gρ':>4}  "
           f"{'dicho Δρ closure refs imp t  visited':<40}  |  "
           f"{'signif Δρ closure refs imp t  visited':<40}")
    print(hdr)
    print("-" * len(hdr))

    sum_d = {"delta": 0, "cert": 0, "t": 0.0, "vis": 0}
    sum_s = {"delta": 0, "cert": 0, "t": 0.0, "vis": 0}
    for k, c_star, reason_json in rows:
        icf = {int(f): tuple(v) for f, v in json.loads(reason_json).items()}
        x = [float(v) for v in ds["X_test"][k]]

        d = chain_once(irf, int(c_star), x, icf, args.cap_s, args.max_nodes, False)
        s = chain_once(irf, int(c_star), x, icf, args.cap_s, args.max_nodes, True)
        sum_d["delta"] += d["delta_rho"]; sum_d["t"] += d["elapsed_s"]; sum_d["vis"] += d["n_visited"]
        sum_d["cert"] += (d["closure"] == "exhausted")
        sum_s["delta"] += s["delta_rho"]; sum_s["t"] += s["elapsed_s"]; sum_s["vis"] += s["n_visited"]
        sum_s["cert"] += (s["closure"] == "exhausted")
        def _fmt(r):
            closure_short = {"exhausted": "CERT", "cap_hit_time": "tcap",
                             "cap_hit_nodes": "ncap", "cap_hit_both": "bcap"}.get(r["closure"], r["closure"][:4])
            return (f"+{r['delta_rho']:>3d}/{r['rho_end']:>4d} "
                    f"{closure_short} ref={r['n_attempts']} imp={r['n_imp']} "
                    f"t={r['elapsed_s']:>5.1f}s nv={r['n_visited']:>5d}")
        print(f"{k:>4}  {d['rho_start']:>4d}  {_fmt(d):<40}  |  {_fmt(s):<40}")

    print()
    n = len(rows)
    print(f"--- aggregate over {n} samples ---")
    print(f"  dichotomic:  Σ Δρ={sum_d['delta']:>4d}  cert={sum_d['cert']}/{n}  "
          f"Σ t={sum_d['t']:.1f}s  Σ visited={sum_d['vis']}")
    print(f"  significant: Σ Δρ={sum_s['delta']:>4d}  cert={sum_s['cert']}/{n}  "
          f"Σ t={sum_s['t']:.1f}s  Σ visited={sum_s['vis']}")


if __name__ == "__main__":
    main()
