#!/usr/bin/env python
"""Greedy maximal reasons sweep across the 66 included datasets.

For each (dataset, sample) the script:

  1. Builds the trivial ICF (slackless, the tightest reason).
  2. Greedy-widens it one cell at a time on either side of every feature,
     committing each move that keeps ``Adv_max → Good`` (the verifier's
     "Good" certification — every completion in the corridor votes c⋆).
  3. Repeats until a full pass finds no more widenings.

What gets saved:

  * ``code/sweeps/maximal_reasons/<dataset>.parquet``
        one row per sample with the full reason (as JSON of
        feature_idx → (b_threshold, e_threshold)) plus the sample values.
  * ``code/sweeps/maximal_reasons/summary.csv``
        per-dataset rollup: n_samples, mean n_constrained, mean widening
        steps, mean elapsed_s.

Reasons are computed in EU-position form (the canonical representation in
this codebase) and additionally serialised in human-readable threshold form.
"""
from __future__ import annotations

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

from drifts.icf import trivial_icf, icf_human
from drifts.milp_majority import majority_check
from drifts.partial_assignment import PartialAssignment
from drifts.profile import forest_profile
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


MODELS = REPO / "models"
ORDER = REPO / "experiments_order" / "included_topo.csv"
OUT_DIR = REPO / "sweeps" / "maximal_reasons"
OUT_SUMMARY = OUT_DIR / "summary.csv"
OUT_DB = OUT_DIR / "sweep.db"

# N_SAMPLES = 0 (or unset) processes every test sample of each dataset.
_RAW_N = os.environ.get("REASONS_N_SAMPLES", "0")
N_SAMPLES = 0 if str(_RAW_N).lower() in ("0", "none", "all", "") else int(_RAW_N)


# ---------- SQLite layout ---------------------------------------------------


SCHEMA = """
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
    inserted_at             TEXT
);
"""


def _open_db() -> sqlite3.Connection:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUT_DB)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _row_parquet_bytes(row: dict) -> bytes:
    """Serialise one row to a single-row parquet payload."""
    buf = io.BytesIO()
    pd.DataFrame([row]).to_parquet(buf, index=False)
    return buf.getvalue()


def _insert_sample(conn: sqlite3.Connection, row: dict) -> None:
    parquet_blob = _row_parquet_bytes(row)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT OR REPLACE INTO reasons (
            dataset, sample, c_star_label, c_star_idx,
            n_features, n_constrained, n_unconstrained,
            n_widening_steps, n_passes, elapsed_s,
            reason_pos_json, reason_threshold_json, sample_x_json,
            sample_parquet, inserted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["dataset"], row["sample"], row["c_star_label"], row["c_star_idx"],
            row["n_features"], row["n_constrained"], row["n_unconstrained"],
            row["n_widening_steps"], row["n_passes"], row["elapsed_s"],
            row["reason_pos_json"], row["reason_threshold_json"], row["sample_x_json"],
            parquet_blob, now,
        ),
    )
    conn.commit()


def _insert_dataset_summary(conn: sqlite3.Connection, row: dict) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT OR REPLACE INTO dataset_summary (
            dataset, n_trees, n_features, n_labels, n_samples,
            mean_n_constrained, mean_n_unconstrained,
            mean_widening_steps, mean_elapsed_s, total_elapsed_s,
            inserted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["dataset"], row["n_trees"], row["n_features"], row["n_labels"],
            row["n_samples"], row["mean_n_constrained"],
            row["mean_n_unconstrained"], row["mean_widening_steps"],
            row["mean_elapsed_s"], row["total_elapsed_s"], now,
        ),
    )
    conn.commit()


def _datasets():
    with ORDER.open() as f:
        return [r["dataset"] for r in csv.DictReader(f) if r.get("dataset")]


def _is_good(irf, n_leaves, c_star: int, icf) -> bool:
    profile = forest_profile(irf, icf)
    alpha = PartialAssignment.initial_from_profile(profile, n_leaves)
    return majority_check(irf, profile, alpha, c_star).verdict == "good"


def greedy_maximal_reason(irf, n_leaves, c_star: int, trivial):
    """Return ``(reason_icf, n_widens, n_passes)``. The reason is the
    fixpoint of one-cell widenings that preserve ``Adv_max → Good``.
    """
    icf = dict(trivial)
    n_widens = 0
    passes = 0
    while True:
        passes += 1
        improved = False
        for i in range(irf.n_features):
            n = len(irf.EU[i])
            # widen left
            b, e = icf[i]
            if b > -1:
                cand = dict(icf); cand[i] = (b - 1, e)
                if _is_good(irf, n_leaves, c_star, cand):
                    icf = cand; n_widens += 1; improved = True
            # widen right
            b, e = icf[i]
            if e < n:
                cand = dict(icf); cand[i] = (b, e + 1)
                if _is_good(irf, n_leaves, c_star, cand):
                    icf = cand; n_widens += 1; improved = True
        if not improved:
            break
    return icf, n_widens, passes


def sweep_one(name: str, conn: sqlite3.Connection) -> dict:
    t0 = time.time()
    rf = joblib.load(MODELS / f"{name}.joblib")["model"]
    irf = from_sklearn(rf, dataset=name)
    def3 = Def3Forest(rf)
    ds = load_dataset(name)
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    n_leaves = irf.n_leaves_per_tree()

    fwd = def3.predict(ds["X_test"])
    n_check = ds["n_test"] if N_SAMPLES == 0 else min(N_SAMPLES, ds["n_test"])
    rows = []
    last_print = time.time()
    for k in range(n_check):
        x = [float(v) for v in ds["X_test"][k]]
        c_star = irf.phi_L[str(fwd[k])]
        trivial = trivial_icf(irf, x)

        t_s = time.time()
        reason, n_widens, n_passes = greedy_maximal_reason(
            irf, n_leaves, c_star, trivial
        )
        t_s = time.time() - t_s

        human = icf_human(reason, irf)
        n_constrained = sum(1 for v in human.values()
                            if v != (None, None))
        n_unconstrained = irf.n_features - n_constrained
        sample_row = {
            "dataset": name,
            "sample": k,
            "c_star_label": inv_phi_L[c_star],
            "c_star_idx": c_star,
            "n_features": irf.n_features,
            "n_constrained": n_constrained,
            "n_unconstrained": n_unconstrained,
            "n_widening_steps": n_widens,
            "n_passes": n_passes,
            "elapsed_s": round(t_s, 3),
            "reason_pos_json": json.dumps(
                {str(i): [b, e] for i, (b, e) in reason.items()}
            ),
            "reason_threshold_json": json.dumps(
                {str(i): [b, e] for i, (b, e) in human.items()}
            ),
            "sample_x_json": json.dumps(x),
        }
        rows.append(sample_row)
        # Persist immediately so the DB tracks the sweep live.
        _insert_sample(conn, sample_row)

        # Live progress for big datasets.
        now = time.time()
        if k == 0 or k == n_check - 1 or (now - last_print) > 10.0:
            print(f"    {name}: {k+1}/{n_check} samples, "
                  f"last n_constrained={n_constrained}/{irf.n_features}, "
                  f"t/sample≈{t_s:.2f}s", flush=True)
            last_print = now

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)

    return {
        "dataset": name,
        "n_trees": irf.n_trees,
        "n_features": irf.n_features,
        "n_labels": irf.n_labels,
        "n_samples": n_check,
        "mean_n_constrained": float(df["n_constrained"].mean()) if rows else 0.0,
        "mean_n_unconstrained": float(df["n_unconstrained"].mean()) if rows else 0.0,
        "mean_widening_steps": float(df["n_widening_steps"].mean()) if rows else 0.0,
        "mean_elapsed_s": float(df["elapsed_s"].mean()) if rows else 0.0,
        "total_elapsed_s": round(time.time() - t0, 2),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = _open_db()
    datasets = _datasets()
    n_str = "ALL" if N_SAMPLES == 0 else str(N_SAMPLES)
    print(f"Maximal reasons sweep, {len(datasets)} datasets, "
          f"N_SAMPLES={n_str} per dataset (full ds['n_test'] when ALL)", flush=True)
    print(f"  DB: {OUT_DB}", flush=True)
    summary = []
    t0 = time.time()
    for i, name in enumerate(datasets, start=1):
        try:
            row = sweep_one(name, conn)
            summary.append(row)
            _insert_dataset_summary(conn, row)
            print(
                f"[{i:>2d}/{len(datasets)}] {name:<30s} "
                f"|constrained|={row['mean_n_constrained']:5.1f}/{row['n_features']:>4d}  "
                f"|widens|={row['mean_widening_steps']:5.1f}  "
                f"t/sample={row['mean_elapsed_s']:.2f}s  "
                f"total={row['total_elapsed_s']:.1f}s",
                flush=True,
            )
        except Exception as e:
            print(f"[{i:>2d}/{len(datasets)}] {name}: CRASH {e!r}", flush=True)
            traceback.print_exc()

    if summary:
        with OUT_SUMMARY.open("w", newline="") as f:
            cols = list(summary[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)

    elapsed = time.time() - t0
    print(f"\n  parquet rows per dataset: {OUT_DIR}/<dataset>.parquet")
    print(f"  SQLite (live, self-contained): {OUT_DB}")
    print(f"    table 'reasons'         — one row per sample (incl. per-sample parquet BLOB)")
    print(f"    table 'dataset_summary' — one row per completed dataset")
    print(f"  summary CSV:              {OUT_SUMMARY}")
    print(f"  total elapsed: {elapsed:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
