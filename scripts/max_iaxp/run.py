"""Compute Max-iAXp explanations on every TEST sample of one UCR dataset.

Loads (or refits, if no joblib cached) the tuned forest from
`metrics/baseline.csv`, builds the RFxpl inputs, then loops over every test
sample calling RFxpl's INFXRF in-process. Results land in:

    results/max-iaxp/<dataset>/
        results.csv      one row per sample
        samples.parquet  raw test arrays
        meta.json        dataset stats, RF params, EU summary, totals
        model.pkl        RFxpl-format pickle (gitignored)
        data.csv         RFxpl data file (gitignored)

Resumable: rows already present in `results.csv` are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `scripts/` importable so `import load_ucr` works regardless of cwd.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# RFxpl uses `namedlist`, which in Python 3.10 needs a `collections.abc` shim.
# Importing this module patches `collections` *before* RFxpl is imported.
from max_iaxp import compat  # noqa: F401  (must precede xrf imports)

import argparse
import contextlib
import csv
import io
import json
import os
import pickle
import re
import signal
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# Per-sample wall-clock cap. Gurobi gets the same value as TimeLimit; SIGALRM
# is a backup in case other phases (encoding, MaxSAT) hang outside Gurobi.
TIMEOUT_S = int(os.environ.get("MAX_IAXP_TIMEOUT_S", "60"))

from load_ucr import load_dataset  # already on sys.path via run script
from max_iaxp.coverage import (
    Interval,
    constrained_feature_ids,
    coverage_convB,
    extract_eu,
    fsc_s,
    parse_explanation_string,
)
from max_iaxp.rfxpl_io import write_data_csv, write_model_pkl


from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
RESULTS_ROOT = REPO_ROOT / "max-iaxp"
MODELS_DIR = REPO_ROOT / "models"
BASELINE_CSV = REPO_ROOT / "metrics" / "baseline.csv"

# RFxpl lives outside this repo; users can override with the env var.
DEFAULT_RFXPL = Path(os.environ.get("RFXPL_PATH", REPO_ROOT.parent.parent / "RFxpl"))


def _add_rfxpl_to_path(rfxpl_path: Path) -> None:
    rfxpl_path = rfxpl_path.resolve()
    if not rfxpl_path.exists():
        sys.exit(f"RFXPL_PATH does not exist: {rfxpl_path}")
    for sub in (rfxpl_path, rfxpl_path / "infxp", rfxpl_path / "xrf"):
        sub_s = str(sub)
        if sub_s not in sys.path:
            sys.path.insert(0, sub_s)


class _InstanceTimeout(Exception):
    pass


def _alarm_handler(_signum, _frame):
    raise _InstanceTimeout


@contextlib.contextmanager
def _hard_wall_timeout(seconds: int):
    if seconds <= 0:
        yield
        return
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


_EXPL_RE = re.compile(r"^\s*expl:\s*(.+)$", re.M)
_COV_RE = re.compile(r"^\s*cov:\s*([0-9.+\-eE]+)\s*$", re.M)
_TIME_RE = re.compile(r"^\s*time:\s*([0-9.+\-eE]+)\s*$", re.M)


def _parse_captured(text: str) -> tuple[str, Optional[float], Optional[float]]:
    expl_match = _EXPL_RE.search(text)
    cov_match = _COV_RE.search(text)
    time_match = _TIME_RE.search(text)
    expl = expl_match.group(1).strip().strip('"') if expl_match else ""
    their = float(cov_match.group(1)) if cov_match else None
    cpu = float(time_match.group(1)) if time_match else None
    return expl, their, cpu


def _refit_rf_from_csv(name: str):
    """Look up best_params for `name` in baseline.csv and refit the forest."""
    from sklearn.ensemble import RandomForestClassifier

    with BASELINE_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["dataset"] == name:
                break
        else:
            sys.exit(f"{name} not in baseline.csv — run scripts/run_all.py first")
    params = json.loads(row["best_params"])
    ds = load_dataset(name)
    final_params = {**params, "criterion": "gini", "random_state": 0, "n_jobs": -1}
    clf = RandomForestClassifier(**final_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(ds["X_train"], ds["y_train"])
    return clf, ds


def _load_rf(name: str):
    joblib_path = MODELS_DIR / f"{name}.joblib"
    if joblib_path.exists():
        import joblib

        bundle = joblib.load(joblib_path)
        rf = bundle["model"]
        ds = load_dataset(name)
        return rf, ds
    return _refit_rf_from_csv(name)


RESULT_FIELDS = [
    "sample_idx",
    "y_true",
    "y_pred",
    "pred_correct",
    "axp_json",
    "intervals_json",
    "n_constrained",
    "coverage_convB",
    "fsc_s",
    "their_cov",
    "solver_status",
    "solver_wall_s",
    "total_wall_s",
    "timeout_s",
]


def _read_latest(results_csv: Path) -> dict[int, dict]:
    """Return {sample_idx: latest_row_dict}. CSV is append-only; later rows win."""
    if not results_csv.exists():
        return {}
    latest: dict[int, dict] = {}
    with results_csv.open() as f:
        for r in csv.DictReader(f):
            if not r.get("sample_idx"):
                continue
            latest[int(r["sample_idx"])] = r
    return latest


def _is_complete_at_timeout(row: dict, timeout_s: float) -> bool:
    """A sample is 'done' for this timeout if it succeeded, is a definitive
    no-explanation, or is a deterministic error (rerunning won't help). It is
    also done if it previously timed out with an equal-or-larger timeout (no
    point retrying)."""
    status = (row or {}).get("solver_status", "")
    if status in ("ok", "no_expl"):
        return True
    if status.startswith("error:"):
        return True
    if status == "timeout":
        prev = float(row.get("timeout_s") or 0.0)
        return prev >= timeout_s
    return False


def _append_row(results_csv: Path, row: dict) -> None:
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def _migrate_schema(results_csv: Path) -> None:
    """If an older results.csv exists with a subset of RESULT_FIELDS as header,
    rewrite it with the current schema (preserving rows; new fields empty)."""
    if not results_csv.exists():
        return
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)
    if set(RESULT_FIELDS) <= set(existing):
        return
    with results_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})


def _write_samples_parquet(out_dir: Path, X_test: np.ndarray) -> Path:
    cols = {"sample_idx": np.arange(len(X_test))}
    cols.update({f"x_{i}": X_test[:, i].astype(np.float32) for i in range(X_test.shape[1])})
    df = pd.DataFrame(cols)
    path = out_dir / "samples.parquet"
    df.to_parquet(path, index=False)
    return path


def _sample_for_rfxpl(x: np.ndarray) -> np.ndarray:
    """INFXRF.explain expects a 1-d numeric numpy array (not a CSV string)."""
    return np.asarray(x, dtype=np.float64)


def _write_meta(out_dir: Path, ds, rf, eu, totals: dict) -> None:
    meta = {
        "dataset": ds["name"],
        "n_train": ds["n_train"],
        "n_test": ds["n_test"],
        "length": ds["length"],
        "n_classes": ds["n_classes"],
        "rf_params": {
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "min_samples_leaf": rf.min_samples_leaf,
            "max_features": rf.max_features,
            "class_weight": rf.class_weight,
            "random_state": rf.random_state,
        },
        "eu_sizes": {str(i): len(eu[i]) for i in eu},
        "totals": totals,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def run_dataset(
    name: str,
    rfxpl_path: Path,
    timeout_s: int = TIMEOUT_S,
    stop_on_first_timeout: bool = False,
) -> dict:
    """Run Max-iAXp over the TEST set of `name`. Samples already completed at
    a sufficient timeout are skipped (CSV is read; latest row per sample wins).

    If `stop_on_first_timeout` is set, the loop exits as soon as a sample
    times out — used by the progressive sweep to move to the next dataset.
    """
    _add_rfxpl_to_path(rfxpl_path)
    import gurobipy as gp  # noqa: E402
    gp.setParam("TimeLimit", float(timeout_s))
    gp.setParam("OutputFlag", 0)
    from xrf import RFSklearn, Dataset  # noqa: E402
    from GiAXp import INFXRF  # noqa: E402

    out_dir = RESULTS_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    model_pkl = out_dir / "model.pkl"
    data_csv = out_dir / "data.csv"

    _migrate_schema(results_csv)

    rf, ds = _load_rf(name)
    write_model_pkl(rf, model_pkl)
    _, x_min, x_max = write_data_csv(ds["X_train"], ds["y_train"], data_csv)
    eu = extract_eu(rf)

    cls = RFSklearn(from_file=str(model_pkl))
    rf_data = Dataset(filename=str(data_csv))
    xrf = INFXRF(cls, rf_data.features, rf_data.targets, verb=1)

    _write_samples_parquet(out_dir, ds["X_test"])

    latest = _read_latest(results_csv)
    X_test, y_test = ds["X_test"], ds["y_test"]
    y_pred_all = rf.predict(X_test)

    totals = {"processed": 0, "ok": 0, "no_expl": 0, "timeout": 0,
              "error": 0, "skipped": 0, "wall_s": 0.0,
              "stopped_at_timeout": None, "timeout_s": int(timeout_s)}
    t_start = time.perf_counter()

    for idx in range(len(X_test)):
        if _is_complete_at_timeout(latest.get(idx, {}), timeout_s):
            totals["skipped"] += 1
            continue
        sample_arr = _sample_for_rfxpl(X_test[idx])
        y_true = str(y_test[idx])
        y_pred = str(y_pred_all[idx])

        for attr in ("enc", "x"):
            if hasattr(xrf, attr):
                delattr(xrf, attr)

        buf = io.StringIO()
        status = "ok"
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf):
            try:
                with _hard_wall_timeout(timeout_s):
                    xrf.explain(
                        sample_arr, "abd", "sat",
                        optimal=True, x_bounds=(x_min, x_max),
                    )
            except _InstanceTimeout:
                status = "timeout"
            except Exception as exc:  # noqa: BLE001
                status = f"error:{type(exc).__name__}"
        solver_wall = time.perf_counter() - t0

        expl, their_cov, _cpu = _parse_captured(buf.getvalue())
        if status == "ok" and not expl:
            status = "no_expl"

        intervals = parse_explanation_string(expl, n_features=ds["length"]) if expl else {}
        axp = constrained_feature_ids(intervals) if intervals else []
        cov_B = coverage_convB(intervals, eu) if intervals else float("nan")
        score_s = fsc_s(intervals, eu) if intervals else float("nan")

        row = {
            "sample_idx": idx,
            "y_true": y_true,
            "y_pred": y_pred,
            "pred_correct": y_true == y_pred,
            "axp_json": json.dumps(axp),
            "intervals_json": json.dumps(
                {str(i): iv.to_list() for i, iv in intervals.items() if not iv.is_unconstrained()}
            ),
            "n_constrained": len(axp),
            "coverage_convB": f"{cov_B:.6f}" if not _isnan(cov_B) else "",
            "fsc_s": f"{score_s:.6f}" if not _isnan(score_s) else "",
            "their_cov": f"{their_cov:.6f}" if their_cov is not None else "",
            "solver_status": status,
            "solver_wall_s": f"{solver_wall:.4f}",
            "total_wall_s": f"{time.perf_counter() - t0:.4f}",
            "timeout_s": int(timeout_s),
        }
        _append_row(results_csv, row)
        latest[idx] = row

        totals["processed"] += 1
        if status == "ok":
            totals["ok"] += 1
        elif status == "no_expl":
            totals["no_expl"] += 1
        elif status == "timeout":
            totals["timeout"] = totals.get("timeout", 0) + 1
        else:
            totals["error"] += 1
        totals["wall_s"] += solver_wall

        if totals["processed"] % 10 == 0 or totals["processed"] <= 5 or status != "ok":
            print(
                f"[{name} @{timeout_s}s] {idx + 1}/{len(X_test)} status={status} "
                f"wall={solver_wall:.2f}s cov_B={cov_B:.3f}",
                flush=True,
            )

        if stop_on_first_timeout and status == "timeout":
            totals["stopped_at_timeout"] = idx
            break

    totals["wall_s"] = round(totals["wall_s"], 3)
    totals["total_wall_s_run"] = round(time.perf_counter() - t_start, 3)
    _write_meta(out_dir, ds, rf, eu, totals)
    return totals


def _isnan(x) -> bool:
    try:
        return x != x
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--rfxpl",
        default=str(DEFAULT_RFXPL),
        help=f"path to RFxpl checkout (default: {DEFAULT_RFXPL})",
    )
    parser.add_argument("--timeout-s", type=int, default=TIMEOUT_S)
    parser.add_argument("--stop-on-first-timeout", action="store_true")
    args = parser.parse_args()
    totals = run_dataset(
        args.dataset, Path(args.rfxpl),
        timeout_s=args.timeout_s,
        stop_on_first_timeout=args.stop_on_first_timeout,
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
