#!/usr/bin/env python
"""Greedy maximal reasons using CEGAR-strict ``is_good_strict``.

Replaces ``maximal_reasons_sweep.py``. Same SQLite layout
(``code/sweeps/maximal_reasons/sweep.db``) — per-sample rows + per-dataset
summary — but the Good predicate of the widener is now
``CEGARMajority.is_good_strict`` instead of the conservative
``majority_check``.

CEGAR is sound, so every reason this produces is still a real reason;
because CEGAR also accepts widenings whose conservative-Adv_max
adversaries are cell-incompatible, the resulting reasons are strictly
wider than the conservative greedy's. The infeasibility cache is
maintained per-dataset across all samples of that dataset.

Per-sample row writes happen immediately, so the DB tracks the sweep
live (queriable concurrently by the refinement sweepers).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

from drifts.cegar_majority import CEGARMajority
from drifts.icf import icf_human, trivial_icf
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


MODELS = REPO / "models"
ORDER = REPO / "experiments_order" / "included_topo.csv"
OUT_DIR = REPO / "sweeps" / "maximal_reasons"
OUT_DB = OUT_DIR / "sweep.db"

_RAW_N = os.environ.get("REASONS_N_SAMPLES", "0")
N_SAMPLES = 0 if str(_RAW_N).lower() in ("0", "none", "all", "") else int(_RAW_N)


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS reasons (
    dataset                 TEXT NOT NULL,
    sample                  INTEGER NOT NULL,
    c_star_label            TEXT,
    c_star_idx              INTEGER,
    n_features              INTEGER,
    n_constrained           INTEGER,
    n_unconstrained         INTEGER,
    n_widening_steps        INTEGER,
    n_passes                INTEGER,
    elapsed_s               REAL,
    n_cegar_calls           INTEGER,
    n_no_goods_learned      INTEGER,
    reason_pos_json         TEXT,
    reason_threshold_json   TEXT,
    sample_x_json           TEXT,
    sample_parquet          BLOB,
    inserted_at             TEXT,
    PRIMARY KEY (dataset, sample)
);

CREATE TABLE IF NOT EXISTS dataset_summary (
    dataset                 TEXT PRIMARY KEY,
    n_trees                 INTEGER,
    n_features              INTEGER,
    n_labels                INTEGER,
    n_samples               INTEGER,
    mean_n_constrained      REAL,
    mean_n_unconstrained    REAL,
    mean_widening_steps     REAL,
    mean_elapsed_s          REAL,
    total_elapsed_s         REAL,
    cache_final_size        INTEGER,
    inserted_at             TEXT
);
"""


def _open_db():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUT_DB)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _datasets():
    with ORDER.open() as f:
        return [r["dataset"] for r in csv.DictReader(f) if r.get("dataset")]


def _row_parquet_bytes(row: dict) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame([row]).to_parquet(buf, index=False)
    return buf.getvalue()


def greedy_widen_cegar(cegar, irf, eu_sizes, trivial, c_star):
    """Greedy widener using CEGAR-strict is_good. Returns
    (icf, n_widens, n_passes, n_cegar_calls)."""
    icf = dict(trivial)
    n_widens = 0
    n_passes = 0
    n_cegar = 0
    while True:
        n_passes += 1
        improved = False
        for i in range(irf.n_features):
            n = eu_sizes[i]
            # widen left
            b, e = icf[i]
            if b > -1:
                cand = dict(icf); cand[i] = (b - 1, e)
                n_cegar += 1
                if cegar.is_good_strict(cand, c_star).verdict == "good":
                    icf = cand; n_widens += 1; improved = True
            # widen right
            b, e = icf[i]
            if e < n:
                cand = dict(icf); cand[i] = (b, e + 1)
                n_cegar += 1
                if cegar.is_good_strict(cand, c_star).verdict == "good":
                    icf = cand; n_widens += 1; improved = True
        if not improved:
            break
    return icf, n_widens, n_passes, n_cegar


def sweep_one_sample(state, name: str, k: int, conn):
    """Process one (dataset, sample) using the pre-loaded ``state`` for that
    dataset. ``state[name]`` lazily bootstraps the IRF / OBDD / CEGAR.
    Returns ``True`` if a new row was written; ``False`` if skipped
    (already done or out-of-range).
    """
    s = _ensure_state(state, name)
    if k >= s["n_test"]:
        return False
    if conn.execute(
        "SELECT 1 FROM reasons WHERE dataset=? AND sample=?", (name, k),
    ).fetchone():
        return False
    irf = s["irf"]; cegar = s["cegar"]; ds_obj = s["ds_obj"]
    inv_phi_L = {v: k_ for k_, v in irf.phi_L.items()}

    x = [float(v) for v in ds_obj["X_test"][k]]
    c_star = irf.phi_L[str(s["fwd"][k])]
    trivial = trivial_icf(irf, x)

    t_s = time.time()
    n_cegar_pre = cegar.total_milp_calls
    no_goods_pre = cegar.total_no_goods_learned
    reason, nw, npass, ncegar = greedy_widen_cegar(
        cegar, irf, s["eu_sizes"], trivial, c_star,
    )
    t_s = time.time() - t_s

    human = icf_human(reason, irf)
    nc = sum(1 for v in human.values() if v != (None, None))
    nu = irf.n_features - nc
    row = {
        "dataset": name, "sample": k,
        "c_star_label": inv_phi_L[c_star], "c_star_idx": c_star,
        "n_features": irf.n_features,
        "n_constrained": nc, "n_unconstrained": nu,
        "n_widening_steps": nw, "n_passes": npass,
        "elapsed_s": round(t_s, 4),
        "n_cegar_calls": cegar.total_milp_calls - n_cegar_pre,
        "n_no_goods_learned": cegar.total_no_goods_learned - no_goods_pre,
        "reason_pos_json": json.dumps(
            {str(i): [b, e] for i, (b, e) in reason.items()}
        ),
        "reason_threshold_json": json.dumps(
            {str(i): [b, e] for i, (b, e) in human.items()}
        ),
        "sample_x_json": json.dumps(x),
    }
    blob = _row_parquet_bytes(row)
    conn.execute(
        """
        INSERT OR REPLACE INTO reasons (
            dataset, sample, c_star_label, c_star_idx,
            n_features, n_constrained, n_unconstrained,
            n_widening_steps, n_passes, elapsed_s,
            n_cegar_calls, n_no_goods_learned,
            reason_pos_json, reason_threshold_json, sample_x_json,
            sample_parquet, inserted_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            row["dataset"], row["sample"], row["c_star_label"], row["c_star_idx"],
            row["n_features"], row["n_constrained"], row["n_unconstrained"],
            row["n_widening_steps"], row["n_passes"], row["elapsed_s"],
            row["n_cegar_calls"], row["n_no_goods_learned"],
            row["reason_pos_json"], row["reason_threshold_json"], row["sample_x_json"],
            blob, _now_utc(),
        ),
    )
    conn.commit()
    s["n_constrained_seq"].append(nc)
    s["n_widens_seq"].append(nw)
    s["elapsed_seq"].append(t_s)
    s["last_nc"] = nc
    s["last_t"] = t_s
    s["touched"] += 1
    return True


def _ensure_state(state, name):
    if name in state:
        return state[name]
    rf = joblib.load(MODELS / f"{name}.joblib")["model"]
    irf = from_sklearn(rf, dataset=name)
    def3 = Def3Forest(rf)
    ds_obj = load_dataset(name)
    ctx = OBDDContext.for_forest(irf, encoding="binary")
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    cegar = CEGARMajority(irf=irf, ctx=ctx)
    state[name] = {
        "irf": irf, "ctx": ctx, "cegar": cegar, "def3": def3,
        "ds_obj": ds_obj, "n_test": ds_obj["n_test"],
        "eu_sizes": [len(irf.EU[i]) for i in range(irf.n_features)],
        "fwd": def3.predict(ds_obj["X_test"]),
        "n_constrained_seq": [], "n_widens_seq": [], "elapsed_seq": [],
        "last_nc": None, "last_t": None,
        "touched": 0,
    }
    return state[name]


def _write_dataset_summary(state, name, conn):
    s = state.get(name)
    if not s or not s["n_constrained_seq"]:
        return
    irf = s["irf"]
    conn.execute(
        """
        INSERT OR REPLACE INTO dataset_summary (
            dataset, n_trees, n_features, n_labels, n_samples,
            mean_n_constrained, mean_n_unconstrained,
            mean_widening_steps, mean_elapsed_s, total_elapsed_s,
            cache_final_size, inserted_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            name, irf.n_trees, irf.n_features, irf.n_labels,
            len(s["n_constrained_seq"]),
            sum(s["n_constrained_seq"])/len(s["n_constrained_seq"]),
            irf.n_features - sum(s["n_constrained_seq"])/len(s["n_constrained_seq"]),
            sum(s["n_widens_seq"])/len(s["n_widens_seq"]),
            sum(s["elapsed_seq"])/len(s["elapsed_seq"]),
            round(sum(s["elapsed_seq"]), 2),
            len(s["cegar"].infeasibility_cache),
            _now_utc(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Legacy per-dataset sweep_one is kept for compatibility (not used by main).
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Greedy-CEGAR sweep over the experimental scope, "
                    "dataset-by-dataset in topo order.",
    )
    p.add_argument(
        "--skip-datasets",
        default="",
        help="Comma-separated list of dataset names to skip during this run. "
             "Skipped datasets are NOT marked completed in dataset_summary, "
             "so a subsequent run without the flag will resume them.",
    )
    return p.parse_args(argv)


def main():
    """Dataset-by-dataset in topo order. The CEGAR no-goods accumulate per
    forest (cell-incompatibility is a forest property), so we want to
    process all of a dataset's samples while its no-good cache is still in
    memory and growing. After a dataset finishes we write
    ``dataset_summary`` and free its state.
    """
    args = _parse_args()
    skip_set = {x.strip() for x in args.skip_datasets.split(",") if x.strip()}
    if skip_set:
        print(f"  --skip-datasets active: {sorted(skip_set)}", flush=True)
    conn = _open_db()
    datasets = _datasets()
    # Pre-fetch n_test per dataset (only loads parquet, no IRF/OBDD bootstrap).
    n_test_by_ds = {}
    for ds in datasets:
        try:
            n_test_by_ds[ds] = load_dataset(ds)["n_test"]
        except Exception as e:
            print(f"WARN: cannot pre-load {ds}: {e!r}", flush=True)
            n_test_by_ds[ds] = 0
    max_n = max(n_test_by_ds.values())
    total_samples = sum(n_test_by_ds.values())
    print(f"Greedy-CEGAR dataset-by-dataset sweep:", flush=True)
    print(f"  {len(datasets)} datasets, max n_test = {max_n}, "
          f"total samples = {total_samples}", flush=True)
    print(f"  DB: {OUT_DB}", flush=True)

    # Resume: figure out which (ds, k) pairs are already in the DB.
    done_keys = set()
    for ds_, k_ in conn.execute("SELECT dataset, sample FROM reasons"):
        done_keys.add((ds_, int(k_)))
    print(f"  already-done samples: {len(done_keys)}", flush=True)
    completed_ds = {ds for ds, in conn.execute("SELECT dataset FROM dataset_summary")}

    state = {}
    t_total = time.time()
    n_processed_this_run = 0
    last_print = time.time()
    for di, ds in enumerate(datasets, start=1):
        if ds in completed_ds:
            continue
        if ds in skip_set:
            print(f"  [{di:>2d}/{len(datasets)}] {ds:<30s} SKIPPED (this run only)",
                  flush=True)
            continue
        if n_test_by_ds[ds] == 0:
            continue
        t_ds = time.time()
        for k in range(n_test_by_ds[ds]):
            if (ds, k) in done_keys:
                continue
            try:
                wrote = sweep_one_sample(state, ds, k, conn)
            except Exception as e:
                print(f"  {ds} k={k}: CRASH {e!r}", flush=True)
                traceback.print_exc()
                continue
            if wrote:
                n_processed_this_run += 1
                done_keys.add((ds, k))
                if time.time() - last_print > 10.0:
                    s = state[ds]
                    print(f"  {ds} [{di}/{len(datasets)}] "
                          f"k={k:>5d}/{n_test_by_ds[ds]} "
                          f"nc={s['last_nc']}/{s['irf'].n_features}  "
                          f"t={s['last_t']:.2f}s  "
                          f"cache={len(s['cegar'].infeasibility_cache)}  "
                          f"this-run total={n_processed_this_run}",
                          flush=True)
                    last_print = time.time()
        # Dataset finished — write summary, free state.
        if state.get(ds, {}).get("touched"):
            _write_dataset_summary(state, ds, conn)
        completed_ds.add(ds)
        print(f"[{di:>2d}/{len(datasets)}] {ds:<30s} done  "
              f"n_test={n_test_by_ds[ds]:>5d}  "
              f"t={time.time()-t_ds:.1f}s  "
              f"cache_final={len(state.get(ds, {'cegar': type('x', (), {'infeasibility_cache': set()})()})['cegar'].infeasibility_cache)}",
              flush=True)
        state.pop(ds, None)
    conn.close()
    print(f"\nTotal elapsed: {time.time() - t_total:.1f}s  "
          f"new rows this run: {n_processed_this_run}")


if __name__ == "__main__":
    main()
