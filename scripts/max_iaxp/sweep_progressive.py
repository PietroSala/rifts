"""Progressive-timeout sweep over the Max-iAXp baseline.

Algorithm (as specified):
  1. Datasets are taken in the order given by a CSV in experiments_order/
     (typically the dashboard's `hasse_inverse_topo.csv` — least demanding
     first). If no CSV is present, the global inverse-topological order over
     all 109 datasets and all axes is generated on the fly.
  2. The first pass uses a per-sample timeout of `INITIAL_TIMEOUT_S` (60 s
     by default).
  3. For each dataset, samples are processed in order starting from sample 0;
     as soon as one times out, the sweep jumps to the next dataset.
  4. When the pass finishes, the timeout is doubled and the sweep restarts,
     this time picking up at the first timed-out sample of each dataset.
     `_is_complete_at_timeout` from run.py decides what to skip: samples that
     are `ok` / `no_expl` / definitive `error:*` are skipped forever; samples
     previously timed-out are retried iff the new timeout is strictly larger
     than the one they previously used.
  5. The sweep stops when a full pass produces no new `ok` rows, or when the
     timeout exceeds `MAX_TIMEOUT_S`.

Results are written to `results/max-iaxp/<dataset>/results.csv` along the way
(append-only; the latest row per sample wins on read), so the existing
dashboard can show partial progress.
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
from typing import Optional

from max_iaxp.run import (
    DEFAULT_RFXPL,
    RESULTS_ROOT,
    _is_complete_at_timeout,
    _read_latest,
    run_dataset,
)

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
ORDER_DIR = REPO_ROOT / "experiments_order"
ORDER_CANONICAL = REPO_ROOT / "metrics" / "experiments_order.csv"

INITIAL_TIMEOUT_S = 60
MAX_TIMEOUT_S = 24 * 60 * 60   # safety cap: 1 day
TIMEOUT_MULTIPLIER = 2


def _read_order() -> list[str]:
    """Return dataset names in execution order.

    Resolution order:
      1. `metrics/experiments_order.csv` — the canonical, version-controlled
         order for this paper (committed alongside `baseline.csv`).
      2. First CSV (alphabetical) in `experiments_order/` — historical
         drop zone for dashboard exports.
      3. Generated on the fly: inverse topo over the 109 datasets and all
         demanding-ness axes.
    """
    for path in (ORDER_CANONICAL, *(sorted(p for p in ORDER_DIR.iterdir() if p.suffix == ".csv")
                                    if ORDER_DIR.exists() else ())):
        if path.exists():
            with path.open() as f:
                names = [r["dataset"] for r in csv.DictReader(f) if r.get("dataset")]
            print(f"using order from {path.relative_to(REPO_ROOT)} "
                  f"({len(names)} datasets)", flush=True)
            return names
    # fallback
    print("no order CSV found — generating inverse topo on the fly", flush=True)
    import build_hasse  # type: ignore

    model_stats = json.loads((REPO_ROOT / "metrics" / "model_stats.json").read_text())
    return build_hasse.topological_order_inverse(
        model_stats, list(build_hasse.AXES.keys()),
    )


def _dataset_has_open_work(name: str, timeout_s: int, n_test: int) -> bool:
    """True iff at least one sample in this dataset can be (re)tried at
    `timeout_s` — i.e. one that is not already done at that timeout."""
    results_csv = RESULTS_ROOT / name / "results.csv"
    if not results_csv.exists():
        return True
    latest = _read_latest(results_csv)
    if len(latest) < n_test:
        return True  # some sample never tried
    for idx in range(n_test):
        if not _is_complete_at_timeout(latest.get(idx, {}), timeout_s):
            return True
    return False


def _dataset_n_test(name: str) -> Optional[int]:
    """Pull n_test from model_stats.json so we don't need to load the dataset."""
    ms_path = REPO_ROOT / "metrics" / "model_stats.json"
    if not ms_path.exists():
        return None
    ms = json.loads(ms_path.read_text())
    return ms.get(name, {}).get("n_test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfxpl", default=str(DEFAULT_RFXPL))
    parser.add_argument("--initial-timeout-s", type=int, default=INITIAL_TIMEOUT_S)
    parser.add_argument("--max-timeout-s", type=int, default=MAX_TIMEOUT_S)
    parser.add_argument("--multiplier", type=int, default=TIMEOUT_MULTIPLIER)
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to these datasets (still in CSV order)")
    args = parser.parse_args()

    datasets = _read_order()
    if args.only:
        wanted = set(args.only)
        datasets = [d for d in datasets if d in wanted]

    rfxpl = Path(args.rfxpl)
    timeout_s = args.initial_timeout_s
    pass_idx = 0
    t_global = time.time()

    while timeout_s <= args.max_timeout_s:
        pass_idx += 1
        print(f"\n=== pass {pass_idx} — timeout {timeout_s}s "
              f"({len(datasets)} datasets) ===", flush=True)
        pass_ok = 0
        pass_timeouts = 0
        pass_processed = 0
        any_progress = False
        for i, name in enumerate(datasets, 1):
            n_test = _dataset_n_test(name)
            if n_test is not None and not _dataset_has_open_work(name, timeout_s, n_test):
                print(f"  [{i:3d}/{len(datasets)}] {name:30s} DONE at this timeout",
                      flush=True)
                continue
            print(f"  [{i:3d}/{len(datasets)}] {name:30s} running…", flush=True)
            try:
                totals = run_dataset(
                    name, rfxpl,
                    timeout_s=timeout_s,
                    stop_on_first_timeout=True,
                )
            except Exception:
                print(f"    EXCEPTION on {name}", flush=True)
                traceback.print_exc()
                continue
            pass_processed += totals.get("processed", 0)
            pass_ok += totals.get("ok", 0)
            pass_timeouts += totals.get("timeout", 0)
            if totals.get("ok", 0) > 0:
                any_progress = True
            stopped = totals.get("stopped_at_timeout")
            tag = f" → next pass at sample {stopped}" if stopped is not None else ""
            print(f"    ok={totals.get('ok', 0)} timeout={totals.get('timeout', 0)} "
                  f"error={totals.get('error', 0)} skipped={totals.get('skipped', 0)} "
                  f"wall={totals.get('wall_s', 0):.1f}s{tag}", flush=True)

        elapsed = time.time() - t_global
        print(f"\n--- pass {pass_idx} done: processed={pass_processed} ok={pass_ok} "
              f"timeouts={pass_timeouts} | total elapsed={elapsed:.1f}s ---", flush=True)

        if pass_timeouts == 0:
            print("no timeouts this pass — sweep complete.", flush=True)
            break
        if not any_progress and pass_idx > 1:
            print("no new 'ok' from previous timeouts — stopping (timeout cap reached "
                  "or solver genuinely cannot handle these instances).", flush=True)
            break
        timeout_s *= args.multiplier

    print(f"\ntotal wall time: {time.time() - t_global:.1f}s", flush=True)


if __name__ == "__main__":
    main()
