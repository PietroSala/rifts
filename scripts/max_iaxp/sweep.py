"""Cross-dataset Max-iAXp sweep over the `included` datasets.

A dataset is `included` iff our test accuracy ranks in the top three among
{ours, tsf, 0.95·tsf, 1nn-dtw, 0.95·dtw} — i.e. ours is within 5% of both
TSF and 1NN-DTW (DOE policy, paper_notes.md §8).

Resumable end-to-end: skips datasets whose `results/max-iaxp/<dataset>/meta.json`
is already present, and within each dataset skips test samples already in
results.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import argparse
import csv
import json
import time
import traceback

from max_iaxp.run import RESULTS_ROOT, DEFAULT_RFXPL, run_dataset

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
BASELINE_CSV = REPO_ROOT / "metrics" / "baseline.csv"
REFERENCE_CSV = REPO_ROOT / "metrics" / "reference.csv"


def _rank_dense(values: list[float]) -> list[int]:
    order = sorted(set(values), reverse=True)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return [pos[v] for v in values]


def included_datasets(max_rank: int = 3) -> list[str]:
    ref = {r["dataset"]: r for r in csv.DictReader(REFERENCE_CSV.open())}
    out: list[str] = []
    for r in csv.DictReader(BASELINE_CSV.open()):
        n = r["dataset"]
        if n not in ref:
            continue
        tsf = ref[n]["acc_tsf"]
        dtw = ref[n]["acc_1nn_dtw"]
        if not tsf or not dtw:
            continue
        ours = float(r["test_acc"]); tsf = float(tsf); dtw = float(dtw)
        ours_rank = _rank_dense([ours, tsf, 0.95 * tsf, dtw, 0.95 * dtw])[0]
        if ours_rank <= max_rank:
            out.append(n)
    return out


def _meta_complete(name: str) -> bool:
    """Treat a dataset as done when its meta.json exists."""
    return (RESULTS_ROOT / name / "meta.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rank", type=int, default=3, help="keep up to this rank (1..5)")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    parser.add_argument("--rfxpl", default=str(DEFAULT_RFXPL))
    parser.add_argument("--force", action="store_true", help="re-run datasets already done")
    args = parser.parse_args()

    datasets = included_datasets(max_rank=args.max_rank)
    if args.only:
        wanted = set(args.only)
        datasets = [d for d in datasets if d in wanted]

    print(f"included datasets (rank <= {args.max_rank}): {len(datasets)}", flush=True)
    t0 = time.time()
    for i, name in enumerate(datasets, 1):
        if not args.force and _meta_complete(name):
            print(f"[{i:3d}/{len(datasets)}] {name:30s} SKIP (meta.json present)", flush=True)
            continue
        print(f"[{i:3d}/{len(datasets)}] {name:30s} running…", flush=True)
        try:
            totals = run_dataset(name, Path(args.rfxpl))
            print(f"    {json.dumps(totals)}", flush=True)
        except Exception:
            print(f"    FAILED on {name}", flush=True)
            traceback.print_exc()

    print(f"\nTotal wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
