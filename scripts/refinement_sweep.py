#!/usr/bin/env python
"""Refinement sweep — single-core randomized B&B improvements on top of greedy.

Reads the greedy DB (``code/sweeps/maximal_reasons/sweep.db``) and, per
sample (in greedy order), chains randomized refinement attempts until no
attempt finds an improvement. Each refinement attempt is a random-walk
search over (ICF, locks) nodes:

  * Pop a random open leaf.
  * If its ICF is not Good (Adv_max → fail) → close.
  * Else if ``rho(icf) > bound`` → **first improvement** found → stop and
    return the new (icf, rho, n_visited).
  * Else if ``max_possible_rho(icf, locks) ≤ bound`` → close (B&B prune).
  * Else if no expandable (feat, dir) under the locks → close.
  * Else pick a random expandable (feat, dir) and push two successors:
      - locked  — same ICF, that direction locked (no future widening on it);
      - expanded — ICF widened by one EU cell in that direction, same locks.

A chain stops when a refinement attempt closes without finding an
improvement. The last attempt's reason is the chain maximum.

DB layout (``code/sweeps/refinements/sweep.db``):

  * ``refinements``                — one row per attempt
  * ``refinement_chain_summary``    — one row per (dataset, sample),
    incl. ``final_reason_parquet`` BLOB
  * ``refinement_dataset_summary``  — one row per completed dataset
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sqlite3
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import joblib
import pandas as pd

from _paths import STATE_ROOT as REPO, DATA_ROOT  # noqa: E402

from drifts.icf import trivial_icf
from drifts.milp_majority import majority_check
from drifts.partial_assignment import PartialAssignment
from drifts.profile import forest_profile
from drifts.sklearn_compat import Def3Forest
from drifts.sklearn_io import from_sklearn

from load_ucr import load_dataset  # noqa: E402


MODELS = REPO / "models"
GREEDY_DB = REPO / "sweeps" / "maximal_reasons" / "sweep.db"
OUT_DIR = REPO / "sweeps" / "refinements"
OUT_DB = OUT_DIR / "sweep.db"

def _maybe_int(name, default):
    v = os.environ.get(name, default)
    if v is None or str(v).lower() in ("none", "null", ""):
        return None
    return int(v)


def _maybe_float(name, default):
    v = os.environ.get(name, default)
    if v is None or str(v).lower() in ("none", "null", ""):
        return None
    return float(v)


# Either cap (or both) may be None to disable that dimension. With both set,
# they OR-combine — refine_once stops when EITHER threshold fires.
MAX_NODES = _maybe_int("REFINEMENT_MAX_NODES", "5000")
MAX_TIME_S = _maybe_float("REFINEMENT_MAX_TIME_S", "60.0")
RNG_SEED = int(os.environ.get("REFINEMENT_SEED", "0"))
USE_SIGNIFICANT = os.environ.get("REFINEMENT_USE_SIGNIFICANT", "0") in ("1", "true", "True")
USE_CEGAR       = os.environ.get("REFINEMENT_USE_CEGAR", "0") in ("1", "true", "True")


# ---------- SQLite -----------------------------------------------------------


SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS refinements (
    dataset                 TEXT NOT NULL,
    sample                  INTEGER NOT NULL,
    refinement_idx          INTEGER NOT NULL,
    started_rho             INTEGER,
    found_rho               INTEGER,
    improvement             INTEGER,         -- 1 / 0
    closure_reason          TEXT,            -- 'improvement_found' | 'improvement_at_max' | 'exhausted' | 'cap_hit'
    n_nodes_visited         INTEGER,
    n_closed_bad            INTEGER,         -- leaves closed because Adv_max said Bad
    n_closed_bb_pruned      INTEGER,         -- leaves closed by max_possible <= bound
    n_closed_saturated      INTEGER,         -- leaves closed because no expandable direction
    n_closed_max_eq         INTEGER,         -- leaves closed because profile==max_profile AND max is Bad
    n_g_cache_hits          INTEGER,         -- profile-G hits (forward dominance)
    n_b_cache_hits          INTEGER,         -- profile-B hits (reverse dominance)
    n_open_leaves           INTEGER,         -- # leaves still on the stack at end of this attempt
    open_leaves_json        TEXT,            -- serialised leaves (resume capability)
    elapsed_s               REAL,
    reason_pos_json         TEXT,
    reason_threshold_json   TEXT,
    n_constrained           INTEGER,
    is_final_max            INTEGER,         -- 1 for the last attempt in the chain
    certified_maximum       INTEGER,         -- 1 iff the chain's final attempt is 'exhausted'
    inserted_at             TEXT,
    PRIMARY KEY (dataset, sample, refinement_idx)
);

CREATE TABLE IF NOT EXISTS refinement_chain_summary (
    dataset                 TEXT NOT NULL,
    sample                  INTEGER NOT NULL,
    greedy_rho              INTEGER,
    final_max_rho           INTEGER,
    n_refinements           INTEGER,
    n_improvements          INTEGER,
    certified_maximum       INTEGER,         -- 1 iff the chain ended on 'exhausted'
    total_elapsed_s         REAL,
    final_reason_parquet    BLOB,
    inserted_at             TEXT,
    PRIMARY KEY (dataset, sample)
);

CREATE TABLE IF NOT EXISTS refinement_dataset_summary (
    dataset                     TEXT PRIMARY KEY,
    n_samples                   INTEGER,
    mean_greedy_rho             REAL,
    mean_final_max_rho          REAL,
    mean_improvement            REAL,
    mean_n_refinements          REAL,
    total_elapsed_s             REAL,
    inserted_at                 TEXT
);
"""


def _open_db() -> sqlite3.Connection:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUT_DB)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- Refinement primitives -------------------------------------------


ICFDict = Dict[int, Tuple[int, int]]


def _rho(icf: ICFDict) -> int:
    return sum(e - b for f, (b, e) in icf.items())


def _is_good(irf, c_star: int, icf: ICFDict, n_leaves) -> bool:
    profile = forest_profile(irf, icf)
    alpha = PartialAssignment.initial_from_profile(profile, n_leaves)
    return majority_check(irf, profile, alpha, c_star).verdict == "good"


def _max_possible(icf: ICFDict, lo_bound: Dict[int, int],
                  hi_bound: Dict[int, int]) -> int:
    """Maximum ``rho`` achievable from this ``icf`` by extending every
    direction to its (possibly tightened) bound."""
    return sum(hi_bound[f] - lo_bound[f] for f in icf)


def _profile_tuple(profile) -> tuple:
    return tuple(frozenset(p) for p in profile)


def _serialise_leaves(leaves) -> str:
    out = []
    for icf, lo, hi, pt in leaves:
        out.append({
            "icf": [[int(f), int(b), int(e)] for f, (b, e) in icf.items()],
            "lo":  [[int(f), int(v)] for f, v in lo.items()],
            "hi":  [[int(f), int(v)] for f, v in hi.items()],
            "pt":  None if pt is None else [sorted(map(int, s)) for s in pt],
        })
    return json.dumps(out, separators=(",", ":"))


def _deserialise_leaves(blob: str):
    out = []
    for item in json.loads(blob):
        icf = {int(t[0]): (int(t[1]), int(t[2])) for t in item["icf"]}
        lo = {int(t[0]): int(t[1]) for t in item["lo"]}
        hi = {int(t[0]): int(t[1]) for t in item["hi"]}
        pt = None if item["pt"] is None else tuple(frozenset(s) for s in item["pt"])
        out.append((icf, lo, hi, pt))
    return out


# ---------- Significant-test branching helpers ------------------------------


def _reachable_classes(tree, icf_outer: Dict[int, Tuple[int, int]], eu_sizes):
    """For every node reachable under ``icf_outer``, the set of class labels
    reachable from that node. Bottom-up memo keyed by ``id(node)``."""
    cache = {}

    def rec(node):
        nid = id(node)
        if nid in cache:
            return cache[nid]
        if node["type"] == "leaf":
            res = frozenset({node["label_idx"]})
        else:
            f = node["feature_idx"]
            p = node["threshold_pos"]
            b, e = icf_outer.get(f, (-1, eu_sizes[f]))
            res_set = set()
            if p > b:
                res_set |= rec(node["low"])
            if p < e:
                res_set |= rec(node["high"])
            res = frozenset(res_set)
        cache[nid] = res
        return res

    rec(tree)
    return cache


def _significant_tests(irf, icf_inner: Dict[int, Tuple[int, int]],
                       icf_outer: Dict[int, Tuple[int, int]],
                       eu_sizes):
    """Per-feature list of significant tests ``(position, direction)``.

    A test ``(f, p)`` is significant iff:
      * the corresponding tree node is reachable under ``icf_outer``;
      * ``icf_inner`` does NOT split on it (``e_in ≤ p`` or ``b_in ≥ p``);
      * ``icf_outer`` DOES split on it (``b_out < p < e_out``);
      * under ``icf_outer``, the classes reachable from the two children
        together cover at least 2 distinct labels.
    """
    out: Dict[int, list] = {}
    for tree in irf.trees:
        outer_classes = _reachable_classes(tree, icf_outer, eu_sizes)
        _walk_significant(tree, icf_inner, icf_outer, outer_classes,
                          eu_sizes, out)
    return out


def _walk_significant(node, icf_inner, icf_outer, outer_classes,
                      eu_sizes, out):
    if node["type"] == "leaf":
        return
    f = node["feature_idx"]
    p = node["threshold_pos"]
    b_in, e_in = icf_inner.get(f, (-1, eu_sizes[f]))
    b_out, e_out = icf_outer.get(f, (-1, eu_sizes[f]))
    outer_splits = (b_out < p) and (p < e_out)
    if outer_splits:
        if e_in <= p:
            inner_dir = "low"; inner_splits = False
        elif b_in >= p:
            inner_dir = "high"; inner_splits = False
        else:
            inner_dir = None;  inner_splits = True
        if not inner_splits:
            left_classes = outer_classes.get(id(node["low"]), frozenset())
            right_classes = outer_classes.get(id(node["high"]), frozenset())
            if len(left_classes | right_classes) >= 2:
                out.setdefault(f, []).append((p, inner_dir))
    # Recurse only into the children reachable under icf_outer.
    if p > b_out:
        _walk_significant(node["low"], icf_inner, icf_outer, outer_classes,
                          eu_sizes, out)
    if p < e_out:
        _walk_significant(node["high"], icf_inner, icf_outer, outer_classes,
                          eu_sizes, out)


def _pick_significant_split(sig_tests):
    """Pick (feature, position, direction) deterministically.

    Feature: maximum number of significant tests (tie-break by lowest
    feature index). Test on that feature: median by position (lower-middle
    on even counts).
    """
    if not sig_tests:
        return None
    f_best = max(sig_tests.keys(),
                 key=lambda f: (len(sig_tests[f]), -f))
    tests = sorted(sig_tests[f_best])
    n = len(tests)
    median = tests[n // 2] if n % 2 == 1 else tests[n // 2 - 1]
    return (f_best, median[0], median[1])


def refine_once(irf, c_star: int, n_leaves, eu_sizes: Sequence[int],
                initial_icf: ICFDict, bound: int,
                rng: random.Random,
                max_nodes=MAX_NODES, max_time_s=MAX_TIME_S,
                initial_leaves=None, g_cache=None, b_cache=None,
                cegar=None):
    """Dichotomic B&B for the *first* improvement over ``bound``.

    At each leaf we pick the ``(feature, direction)`` with the **largest
    free-cells-to-bound** count (random tiebreak) and split:

      * **Expand**:  ICF endpoint jumps by ⌈free/2⌉ cells in that direction,
                      bounds unchanged. New ICF — must re-check is_good.
      * **Tighten**: ICF endpoint stays put, but the bound on that side is
                      pulled one cell below the midpoint. ICF unchanged
                      from parent → is_good inherited.

    Closure: not-Good, ``max_possible ≤ bound`` (B&B prune), or every
    direction saturated (``hi == e`` and ``lo == b`` for every f).

    Returns ``(icf, rho, n_visited, "improvement_found")`` on first
    improvement, or ``(None, bound, n_visited, "exhausted" | "cap_hit")``
    otherwise. ``"exhausted"`` means the leaves list emptied — the search
    is complete and ``bound`` is the certified maximum. ``"cap_hit"``
    means the search ran out of ``max_nodes`` budget.
    """
    # G / B profile caches (antichains). Carried across chain attempts so
    # later attempts inherit the work of earlier ones.
    if g_cache is None:
        g_cache = []     # tuples-of-frozensets; antichain of MOST GENERAL profiles
    if b_cache is None:
        b_cache = []     # tuples-of-frozensets; antichain of MOST SPECIFIC profiles
    counters = {"bad": 0, "bb_pruned": 0, "saturated": 0,
                "max_eq": 0, "g_hits": 0, "b_hits": 0}

    def _g_hit(pt):
        return any(all(pt[t] <= c[t] for t in range(len(c))) for c in g_cache)

    def _b_hit(pt):
        return any(all(c[t] <= pt[t] for t in range(len(c))) for c in b_cache)

    def _g_insert(pt):
        if _g_hit(pt):
            return
        g_cache[:] = [c for c in g_cache
                      if not all(c[t] <= pt[t] for t in range(len(c)))]
        g_cache.append(pt)

    def _b_insert(pt):
        if _b_hit(pt):
            return
        b_cache[:] = [c for c in b_cache
                      if not all(pt[t] <= c[t] for t in range(len(c)))]
        b_cache.append(pt)

    def _verdict(icf):
        """Returns (is_good_bool, profile_tuple). Uses + populates caches.

        When the worker passes a ``cegar`` instance (CEGAR strict path), we
        run ``is_good_strict`` instead of the conservative ``majority_check``
        for the actual verdict; profile-tuple membership caches are still
        consulted first to skip the expensive call.
        """
        profile = forest_profile(irf, icf)
        pt = _profile_tuple(profile)
        if _g_hit(pt):
            counters["g_hits"] += 1
            return True, pt
        if _b_hit(pt):
            counters["b_hits"] += 1
            return False, pt
        if cegar is not None:
            verdict_str = cegar.is_good_strict(icf, c_star).verdict
            is_good = (verdict_str == "good")
        else:
            alpha = PartialAssignment.initial_from_profile(profile, n_leaves)
            is_good = majority_check(irf, profile, alpha, c_star).verdict == "good"
        if is_good:
            _g_insert(pt)
            return True, pt
        _b_insert(pt)
        return False, pt

    # If the caller passed remaining leaves from a previous attempt, restart
    # from those; otherwise seed with the trivial root.
    if initial_leaves is not None:
        leaves = list(initial_leaves)
    else:
        initial_lo = {f: -1 for f in initial_icf}
        initial_hi = {f: eu_sizes[f] for f in initial_icf}
        # node tuple = (icf, lo, hi, profile_tuple_or_None)
        leaves = [(dict(initial_icf), initial_lo, initial_hi, None)]
    n_visited = 0
    t_start = time.time()

    def _node_cap_hit():
        return max_nodes is not None and n_visited >= max_nodes

    def _time_cap_hit():
        return max_time_s is not None and (time.time() - t_start) >= max_time_s

    while leaves and not _node_cap_hit() and not _time_cap_hit():
        idx = rng.randint(0, len(leaves) - 1)
        icf, lo, hi, pt = leaves.pop(idx)
        n_visited += 1

        # If we don't know the profile/verdict yet, compute it. Leaves
        # inherited from a previous attempt may carry a stale profile flag
        # but it is still consistent (profile depends only on ICF), so
        # `pt is None` here means "compute is_good now".
        if pt is None:
            ok, pt = _verdict(icf)
            if not ok:
                counters["bad"] += 1
                continue
        # At this point icf is Good with profile pt.

        r = _rho(icf)
        if r > bound:
            return icf, r, n_visited, "improvement_found", counters, leaves, g_cache, b_cache

        max_pos = _max_possible(icf, lo, hi)
        if max_pos <= bound:
            counters["bb_pruned"] += 1
            continue

        # --- max-inflated check ----------------------------------------
        max_icf = {f: (lo[f], hi[f]) for f in icf}
        if max_icf == icf:
            # Already saturated under the bounds; rho(icf) IS the max
            # achievable here, and we know rho <= bound (else returned above).
            counters["saturated"] += 1
            continue
        max_is_good, max_pt = _verdict(max_icf)
        if max_is_good:
            # Guaranteed improvement at the max-inflated icf.
            return max_icf, max_pos, n_visited, "improvement_at_max", counters, leaves, g_cache, b_cache
        if pt == max_pt:
            # Same profile across the whole subtree → all intermediates Bad.
            counters["max_eq"] += 1
            continue
        # ---------------------------------------------------------------

        if USE_SIGNIFICANT:
            # Branch on a class-discriminative test rather than on max-free-cells.
            outer = {f: (lo[f], hi[f]) for f in icf}
            sig = _significant_tests(irf, icf, outer, eu_sizes)
            pick = _pick_significant_split(sig)
            if pick is None:
                # No class-discriminating widening admissible — close.
                counters["saturated"] += 1
                continue
            f, p, direction = pick
            if direction == "high":
                # icf is on the high side of p (b ≥ p), outer splits.
                # Successor A: lift outer's lower bound to p (preserve high
                # turn — outer no longer crosses the test).
                lo_a = dict(lo); lo_a[f] = p
                leaves.append((dict(icf), lo_a, dict(hi), pt))
                # Successor B: lower icf's b to p-1 (force the inner to
                # split at this test, opening both children).
                icf_b = dict(icf); icf_b[f] = (p - 1, icf[f][1])
                leaves.append((icf_b, dict(lo), dict(hi), None))
            else:  # direction == "low"
                # icf is on the low side of p (e ≤ p), outer splits.
                # Successor A: lower outer's upper bound to p (preserve low
                # turn — outer no longer crosses the test).
                hi_a = dict(hi); hi_a[f] = p
                leaves.append((dict(icf), dict(lo), hi_a, pt))
                # Successor B: lift icf's e to p+1 (force the inner to split).
                icf_b = dict(icf); icf_b[f] = (icf[f][0], p + 1)
                leaves.append((icf_b, dict(lo), dict(hi), None))
            continue

        # Default dichotomic split: pick the (f, d) with the most free cells.
        cands = []
        max_n = 0
        for f, (b, e) in icf.items():
            n_l = b - lo[f]
            n_r = hi[f] - e
            if n_l > max_n:
                max_n = n_l; cands = [(f, "left")]
            elif n_l == max_n and n_l > 0:
                cands.append((f, "left"))
            if n_r > max_n:
                max_n = n_r; cands = [(f, "right")]
            elif n_r == max_n and n_r > 0:
                cands.append((f, "right"))
        if max_n <= 0:
            counters["saturated"] += 1
            continue
        f, d = rng.choice(cands)
        step = (max_n + 1) // 2          # ceil(max_n / 2) ≥ 1

        # Branch A — expand by `step` cells; new ICF, verdict unknown.
        icf_a = dict(icf)
        b, e = icf_a[f]
        if d == "left":
            icf_a[f] = (b - step, e)
        else:
            icf_a[f] = (b, e + step)
        leaves.append((icf_a, dict(lo), dict(hi), None))

        # Branch B — tighten the bound; ICF unchanged; profile unchanged,
        # is_good inherited from parent (= True), so carry the profile pt.
        lo_b = dict(lo); hi_b = dict(hi)
        if d == "left":
            lo_b[f] = (b - step) + 1
        else:
            hi_b[f] = (e + step) - 1
        leaves.append((icf, lo_b, hi_b, pt))

    if not leaves:
        closure = "exhausted"
    else:
        n_hit = _node_cap_hit()
        t_hit = _time_cap_hit()
        if n_hit and t_hit:
            closure = "cap_hit_both"
        elif t_hit:
            closure = "cap_hit_time"
        else:
            closure = "cap_hit_nodes"
    return None, bound, n_visited, closure, counters, leaves, g_cache, b_cache


# ---------- per-sample chain -------------------------------------------------


def _row_parquet_bytes(row: dict) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame([row]).to_parquet(buf, index=False)
    return buf.getvalue()


def _human_threshold(icf: ICFDict, irf) -> Dict[str, list]:
    out = {}
    for f, (b, e) in icf.items():
        eu_i = irf.EU[f]
        b_t = None if b < 0 else float(eu_i[b])
        e_t = None if e >= len(eu_i) else float(eu_i[e])
        out[str(f)] = [b_t, e_t]
    return out


def _resume_state(conn: sqlite3.Connection, dataset: str, sample: int,
                  greedy_icf: ICFDict):
    """Determine where to start the chain for this (dataset, sample).

    Returns ``None`` if the chain is already certified maximum — caller should
    skip the sample. Otherwise returns
    ``(current_icf, current_rho, refinement_idx, leaves_state,
       prior_elapsed_s, prior_n_refinements, prior_n_improvements)`` where the
    last three are *cumulative* counters from prior runs (so the new chain
    can add to them and update ``refinement_chain_summary`` consistently).
    """
    chain = conn.execute(
        "SELECT certified_maximum, total_elapsed_s, n_refinements, "
        "n_improvements FROM refinement_chain_summary "
        "WHERE dataset=? AND sample=?", (dataset, sample),
    ).fetchone()
    if chain and chain[0] == 1:
        return None   # certified — nothing to do

    last = conn.execute(
        "SELECT refinement_idx, closure_reason, found_rho, reason_pos_json, "
        "open_leaves_json FROM refinements "
        "WHERE dataset=? AND sample=? "
        "ORDER BY refinement_idx DESC LIMIT 1", (dataset, sample),
    ).fetchone()
    if last is None:
        # Fresh sample.
        return (dict(greedy_icf), _rho(greedy_icf), 0, None, 0.0, 0, 0)

    if last[1] == "exhausted":
        # Already certified by the last attempt; chain_summary should have
        # certified_maximum=1 already, but guard against drift.
        return None

    # Resume from the last cap-hit / improvement row.
    raw = json.loads(last[3])
    current_icf = {int(f): tuple(v) for f, v in raw.items()}
    current_rho = int(last[2])
    refinement_idx = int(last[0]) + 1
    leaves_state = _deserialise_leaves(last[4]) if last[4] else None
    prior_elapsed = float(chain[1]) if chain else 0.0
    prior_n_ref = int(chain[2]) if chain else 0
    prior_n_imp = int(chain[3]) if chain else 0
    return (current_icf, current_rho, refinement_idx, leaves_state,
            prior_elapsed, prior_n_ref, prior_n_imp)


def refine_sample(conn: sqlite3.Connection, irf, c_star: int, x,
                  greedy_icf: ICFDict, dataset: str, sample: int,
                  rng: random.Random,
                  max_nodes=MAX_NODES, max_time_s=MAX_TIME_S,
                  cegar=None):
    """Chain refinement attempts. Each attempt that hits a cap stores its
    remaining leaves so a later invocation (with a different cap / time
    budget) can pick up via ``_resume_state`` and continue *as a new row*
    (no overwriting). The chain ends when an attempt closes ``exhausted`` —
    the current best reason is then the certified chain maximum."""
    n_leaves = irf.n_leaves_per_tree()
    eu_sizes = [len(irf.EU[i]) for i in range(irf.n_features)]
    greedy_rho = _rho(greedy_icf)

    state = _resume_state(conn, dataset, sample, greedy_icf)
    if state is None:
        return None       # caller skips (already certified)
    (current_icf, current_rho, refinement_idx, leaves_state,
     prior_elapsed, prior_n_ref, prior_n_imp) = state

    # Root for every fresh-attempt trivial-seed (used only when leaves_state
    # is None). The bound is `current_rho`, the live max we know so far.
    trivial = trivial_icf(irf, x)

    total_elapsed = 0.0
    n_improvements = 0
    final_closure = "exhausted"  # tentatively
    g_cache: list = []
    b_cache: list = []
    while True:
        t0 = time.time()
        new_icf, found_rho, n_visited, closure, counters, leaves_state, g_cache, b_cache = refine_once(
            irf, c_star, n_leaves, eu_sizes,
            trivial, current_rho, rng,
            max_nodes=max_nodes, max_time_s=max_time_s,
            initial_leaves=leaves_state,
            g_cache=g_cache, b_cache=b_cache,
            cegar=cegar,
        )
        elapsed = time.time() - t0
        total_elapsed += elapsed
        n_open = len(leaves_state)
        leaves_blob = _serialise_leaves(leaves_state) if leaves_state else None

        if closure in ("improvement_found", "improvement_at_max"):
            n_constrained = sum(
                1 for f, (b, e) in new_icf.items()
                if not (b < 0 and e >= eu_sizes[f])
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO refinements (
                    dataset, sample, refinement_idx, started_rho, found_rho,
                    improvement, closure_reason, n_nodes_visited,
                    n_closed_bad, n_closed_bb_pruned, n_closed_saturated,
                    n_closed_max_eq, n_g_cache_hits, n_b_cache_hits,
                    n_open_leaves, open_leaves_json,
                    elapsed_s, reason_pos_json, reason_threshold_json,
                    n_constrained, is_final_max, certified_maximum, inserted_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (dataset, sample, refinement_idx, current_rho, found_rho,
                 1, closure, n_visited,
                 counters["bad"], counters["bb_pruned"], counters["saturated"],
                 counters["max_eq"], counters["g_hits"], counters["b_hits"],
                 n_open, leaves_blob,
                 round(elapsed, 4),
                 json.dumps({str(f): [b, e] for f, (b, e) in new_icf.items()}),
                 json.dumps(_human_threshold(new_icf, irf)),
                 n_constrained, 0, 0, _now_utc()),
            )
            conn.commit()
            n_improvements += 1
            current_icf, current_rho = new_icf, found_rho
            refinement_idx += 1
            continue

        # closure ∈ {"exhausted", "cap_hit"} — no improvement this attempt.
        final_closure = closure
        certified = 1 if closure == "exhausted" else 0
        n_constrained = sum(
            1 for f, (b, e) in current_icf.items()
            if not (b < 0 and e >= eu_sizes[f])
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO refinements (
                dataset, sample, refinement_idx, started_rho, found_rho,
                improvement, closure_reason, n_nodes_visited,
                n_closed_bad, n_closed_bb_pruned, n_closed_saturated,
                n_closed_max_eq, n_g_cache_hits, n_b_cache_hits,
                n_open_leaves, open_leaves_json,
                elapsed_s, reason_pos_json, reason_threshold_json,
                n_constrained, is_final_max, certified_maximum, inserted_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (dataset, sample, refinement_idx, current_rho, current_rho,
             0, closure, n_visited,
             counters["bad"], counters["bb_pruned"], counters["saturated"],
             counters["max_eq"], counters["g_hits"], counters["b_hits"],
             n_open, leaves_blob,
             round(elapsed, 4),
             json.dumps({str(f): [b, e] for f, (b, e) in current_icf.items()}),
             json.dumps(_human_threshold(current_icf, irf)),
             n_constrained, 1, certified, _now_utc()),
        )
        conn.commit()
        break

    # Chain summary row + final-reason parquet BLOB
    final_row = {
        "dataset": dataset, "sample": sample,
        "c_star_idx": c_star, "n_features": irf.n_features,
        "rho": current_rho,
        "reason_pos_json": json.dumps(
            {str(f): [b, e] for f, (b, e) in current_icf.items()}
        ),
        "reason_threshold_json": json.dumps(_human_threshold(current_icf, irf)),
        "sample_x_json": json.dumps([float(v) for v in x]),
    }
    blob = _row_parquet_bytes(final_row)
    certified = 1 if final_closure == "exhausted" else 0
    # Cumulative across all invocations:
    cum_refs = prior_n_ref + (refinement_idx + 1 - (prior_n_ref))
    # ``refinement_idx`` already includes prior rows because resume increments
    # past the previous max — so the actual number of stored rows is
    # ``refinement_idx + 1``. Use that directly for n_refinements.
    cum_refs = refinement_idx + 1
    cum_imp = prior_n_imp + n_improvements
    cum_elapsed = round(prior_elapsed + total_elapsed, 4)
    conn.execute(
        """
        INSERT OR REPLACE INTO refinement_chain_summary (
            dataset, sample, greedy_rho, final_max_rho,
            n_refinements, n_improvements, certified_maximum,
            total_elapsed_s, final_reason_parquet, inserted_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (dataset, sample, greedy_rho, current_rho,
         cum_refs, cum_imp, certified,
         cum_elapsed, blob, _now_utc()),
    )
    conn.commit()
    return (greedy_rho, current_rho, cum_refs, cum_imp,
            cum_elapsed, final_closure)


# ---------- main loop --------------------------------------------------------


def _read_greedy_rows():
    """Snapshot read from the greedy DB (rowid order)."""
    if not GREEDY_DB.exists():
        raise FileNotFoundError(GREEDY_DB)
    g = sqlite3.connect(f"file:{GREEDY_DB}?mode=ro", uri=True)
    rows = g.execute(
        "SELECT dataset, sample, c_star_idx, reason_pos_json "
        "FROM reasons ORDER BY rowid"
    ).fetchall()
    g.close()
    return rows


def _already_certified(conn: sqlite3.Connection):
    """Samples whose chain reached ``certified_maximum=1`` — these are done."""
    return {
        (ds, k)
        for ds, k in conn.execute(
            "SELECT dataset, sample FROM refinement_chain_summary "
            "WHERE certified_maximum = 1"
        )
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most this many (dataset, sample) pairs")
    args = ap.parse_args()

    conn = _open_db()
    done = _already_certified(conn)
    rows = _read_greedy_rows()
    todo = [r for r in rows if (r[0], r[1]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Refinement sweep: {len(todo)} sample(s) to chain "
          f"(skipping {len(rows) - len(todo)} already done)", flush=True)
    print(f"  DB: {OUT_DB}", flush=True)

    rng = random.Random(RNG_SEED)
    t0 = time.time()
    by_ds = {}
    for i, (ds, k, c_star, reason_json) in enumerate(todo, start=1):
        try:
            rf = joblib.load(MODELS / f"{ds}.joblib")["model"]
            irf = from_sklearn(rf, dataset=ds)
            x = [float(v) for v in load_dataset(ds)["X_test"][k]]
            raw = json.loads(reason_json)
            greedy_icf = {int(f): tuple(v) for f, v in raw.items()}
            res = refine_sample(
                conn, irf, c_star, x, greedy_icf, ds, k,
                random.Random(RNG_SEED + 10_000 * k + hash(ds) % 10_000),
            )
            if res is None:
                print(f"[{i:>4d}/{len(todo)}] {ds:<28s} k={k}  "
                      f"(already certified, skipping)", flush=True)
                continue
            r0, r1, n_ref, n_imp, t, closure = res
            by_ds.setdefault(ds, []).append((r0, r1, n_ref, n_imp, t))
            row = conn.execute(
                "SELECT SUM(n_closed_bad), SUM(n_closed_bb_pruned), "
                "SUM(n_closed_saturated), SUM(n_closed_max_eq), "
                "SUM(n_g_cache_hits), SUM(n_b_cache_hits) FROM refinements "
                "WHERE dataset=? AND sample=?", (ds, k),
            ).fetchone()
            nb, npb, ns, nm, ng, nbh = [int(x or 0) for x in (row or (0,0,0,0,0,0))]
            cert = "CERTIFIED" if closure == "exhausted" else closure
            print(
                f"[{i:>4d}/{len(todo)}] {ds:<28s} k={k}  "
                f"ρ {r0:>4d} → {r1:>4d}  +{r1 - r0:<3d}  "
                f"refs={n_ref} imp={n_imp} [{cert}]  "
                f"closed: bad={nb} bb={npb} sat={ns} max_eq={nm}  "
                f"hits: G={ng} B={nbh}  "
                f"t={t:.2f}s",
                flush=True,
            )
            # Update dataset summary if this finishes the dataset
            ds_samples = sum(1 for r in rows if r[0] == ds)
            if len(by_ds[ds]) >= ds_samples:
                stats = by_ds[ds]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO refinement_dataset_summary (
                        dataset, n_samples,
                        mean_greedy_rho, mean_final_max_rho, mean_improvement,
                        mean_n_refinements, total_elapsed_s, inserted_at
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (ds, len(stats),
                     sum(s[0] for s in stats)/len(stats),
                     sum(s[1] for s in stats)/len(stats),
                     sum(s[1] - s[0] for s in stats)/len(stats),
                     sum(s[2] for s in stats)/len(stats),
                     sum(s[4] for s in stats),
                     _now_utc()),
                )
                conn.commit()
        except Exception as e:
            print(f"[{i:>4d}/{len(todo)}] {ds} k={k}: CRASH {e!r}", flush=True)
            traceback.print_exc()

    conn.close()
    print(f"\n  refinement DB: {OUT_DB}")
    print(f"  total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
