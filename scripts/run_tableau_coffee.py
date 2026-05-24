#!/usr/bin/env python
"""End-to-end tableau B&B run on Coffee, sample 0.

Builds the verifier stack (IRF, OBDDContext, caches), seeds the root cICF
``(trivial_icf, trivial_icf, free_ceiling)``, then runs a single
``TableauWorker`` until the §20 termination predicate fires. Prints the
final incumbent.

Knobs:
  --sample, -k        sample index (default 0)
  --max-loops, -L     cap the loop count (defensive, default 5000)
  --ttl               claim TTL (default 30s)
  --rho               'cells' or 'constrained' (default cells)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

from cache.caches import open_caches, wipe_dataset_caches
from cache.connection import get_client
from drifts.icf import trivial_icf, icf_human
from drifts.obdd import BootstrapConfig, OBDDContext
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn
from tableau import (
    CICF, TableauKeys, TableauWorker, canonical_icf_key,
    make_verifier_adapter, rho_cells, rho_constrained_features,
)

from load_ucr import load_dataset  # noqa: E402


def _wipe_tableau(r, keys: TableauKeys) -> None:
    """Delete every key under the tableau prefix for this (ds, sample)."""
    pattern = f"{keys.prefix}:*"
    batch = []
    for k in r.scan_iter(match=pattern):
        batch.append(k)
        if len(batch) >= 500:
            r.delete(*batch); batch = []
    if batch:
        r.delete(*batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Coffee")
    ap.add_argument("--sample", "-k", type=int, default=0)
    ap.add_argument("--max-loops", "-L", type=int, default=5000)
    ap.add_argument("--ttl", type=int, default=30)
    ap.add_argument("--rho", choices=("cells", "constrained"), default="cells")
    ap.add_argument("--encoding", choices=("binary", "one_hot"), default="binary")
    args = ap.parse_args()

    t0 = time.time()
    rf = joblib.load(REPO / "models" / f"{args.dataset}.joblib")["model"]
    irf = from_sklearn(rf, dataset=args.dataset)
    def3 = Def3Forest(rf)
    ds = load_dataset(args.dataset)
    ctx = OBDDContext.for_forest(irf, encoding=args.encoding)
    ctx.bootstrap(BootstrapConfig(mode="per_worker"))
    eu_sizes = [len(irf.EU[i]) for i in range(irf.n_features)]
    t_build = time.time() - t0

    x = [float(v) for v in ds["X_test"][args.sample]]
    fwd = def3.predict(ds["X_test"])
    label_idx_of = {k: v for k, v in irf.phi_L.items()}
    inv_phi_L = {v: k for k, v in irf.phi_L.items()}
    c_star = label_idx_of[str(fwd[args.sample])]

    print(f"[{args.dataset} k={args.sample}] c⋆ = {inv_phi_L[c_star]} "
          f"({c_star}), n_features={irf.n_features}, n_trees={irf.n_trees}, "
          f"build={t_build:.1f}s, encoding={args.encoding}",
          flush=True)

    trivial = trivial_icf(irf, x)
    free_ceiling = {i: (-1, eu_sizes[i]) for i in range(irf.n_features)}
    root = CICF(inner=dict(trivial), floor=dict(trivial),
                ceiling=free_ceiling, class_label=c_star)

    r = get_client("DATA")
    SESSION = int(time.time() * 1000)
    cache_ds_key = f"_tableau_{args.dataset}_{args.sample}_{SESSION}"
    keys = TableauKeys(dataset=cache_ds_key, sample_idx=args.sample)
    _wipe_tableau(r, keys)
    wipe_dataset_caches(r, cache_ds_key)

    # Conservative verifier adapter (Adv_max → Good on the initial α).
    verify = make_verifier_adapter(irf)

    rho_fn = rho_cells if args.rho == "cells" else rho_constrained_features
    w = TableauWorker(
        r=r, keys=keys, verify=verify, eu_sizes=eu_sizes,
        rho=rho_fn, ttl=args.ttl,
        worker_id=f"tableau-{args.dataset}-{args.sample}",
    )
    w.seed_root(root)

    # Defensive loop cap — overrides run_forever to stop early
    stats = {"verified": 0, "good": 0, "bad": 0, "saturated": 0,
             "bb_pruned": 0, "gc_reaped": 0, "loops": 0}
    t_loop0 = time.time()
    last_print = 0
    while stats["loops"] < args.max_loops:
        stats["loops"] += 1
        w._heartbeat()
        if r.get(keys.state) in (b"done", "done", b"aborted", "aborted"):
            break
        claim = w._claim_top()
        if claim is None:
            if w._termination_predicate():
                r.set(keys.state, "done")
                break
            time.sleep(0.05)
            continue
        node_id, _score = claim
        w._process(node_id, stats)
        if time.time() - last_print > 5.0:
            best = r.hgetall(keys.best) or {}
            best = {k.decode() if isinstance(k, bytes) else k:
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in best.items()}
            n_leaves = r.zcard(keys.leaves)
            print(f"  loop {stats['loops']:>4d}: "
                  f"verified={stats['verified']} good={stats['good']} "
                  f"bad={stats['bad']} pruned={stats['bb_pruned']} "
                  f"saturated={stats['saturated']} "
                  f"|leaves|={n_leaves} best.rho={best.get('rho', '-')}",
                  flush=True)
            last_print = time.time()

    elapsed = time.time() - t_loop0
    print(f"\n[{args.dataset} k={args.sample}] done in {elapsed:.1f}s "
          f"({stats['loops']} loops)", flush=True)
    for k, v in stats.items():
        print(f"  {k:<12s} {v}")

    best = r.hgetall(keys.best) or {}
    best = {k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in best.items()}
    if best:
        print(f"\nIncumbent:")
        print(f"  rho   = {best.get('rho')}")
        print(f"  node  = {best.get('id', '')[:16]}…")
        # decode the inner ICF for a human summary
        icf_pos = {int(k): tuple(v) for k, v in json.loads(best.get("icf", "{}")).items()}
        human = icf_human(icf_pos, irf)
        unconstrained = sum(1 for v in human.values() if v == (None, None))
        constrained = irf.n_features - unconstrained
        print(f"  inner: {constrained}/{irf.n_features} constrained features, "
              f"{unconstrained} unconstrained")
    else:
        print("(no incumbent found)")


if __name__ == "__main__":
    main()
