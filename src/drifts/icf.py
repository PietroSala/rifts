"""Interval Constraint Function (Definitions 4, 9 of `icf-foundations.md`).

ICFs are stored in **EU-index form**: a dict mapping `feature_idx → (b_pos, e_pos)`
where positions are integers over `{-1, 0, 1, …, |EU(i)|}`. The sentinel `-1`
denotes `-∞` (left endpoint outside the EU), and `|EU(i)|` denotes `+∞`.

This is exactly the indexed counterpart of an open-on-the-left,
closed-on-the-right interval `(b, e]` with `b ∈ EU(i) ∪ {-∞}` and
`e ∈ EU(i) ∪ {+∞}` from Definition 9. Storing positions instead of raw
thresholds makes every operation (intersection, extension, domination) integer
arithmetic and is uniform with the Boolean encoding of Part VIII.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .indexed_forest import IndexedRandomForest

# ICF in EU-index form: feature_idx -> (b_pos, e_pos)
ICFIndexed = Dict[int, Tuple[int, int]]


# ---------- trivial ICF (sample → ICF) ---------------------------------------


def trivial_icf(irf: IndexedRandomForest, x: List[float]) -> ICFIndexed:
    """The unique EU-respecting ICF whose interval per feature bracket `x[i]`,
    following the half-open bracket convention `(b, e]` from Definitions 1, 4.

    For feature index `i` with EU sequence `[r_0, r_1, …, r_{k-1}]` and sample
    value `v = x[i]`:
      - `b_pos = max{p : r_p < v}` (strict), or `-1` if `v ≤ r_0`;
      - `e_pos = min{p : v ≤ r_p}` (non-strict), or `k` if `v > r_{k-1}`.

    The resulting interval is `(b, e]` with `b = EU[i][b_pos]` (or `-∞`) and
    `e = EU[i][e_pos]` (or `+∞`). The convention places `v` on the
    right-closed edge when `v` coincides with some `r_p`, matching the test
    `x[f] ≤ r` of Definition 1.
    """
    if len(x) != irf.n_features:
        raise ValueError(
            f"sample has {len(x)} values but forest has {irf.n_features} features"
        )
    icf: ICFIndexed = {}
    for i in range(irf.n_features):
        eu_i = irf.EU[i]
        v = float(x[i])
        b_pos = _last_lt(eu_i, v)            # largest p s.t. EU[p] < v;  -1 if none
        e_pos = _first_ge(eu_i, v)           # smallest p s.t. v <= EU[p]; k if none
        icf[i] = (b_pos, e_pos)
    return icf


def _last_lt(eu: List[str], v: float) -> int:
    """Largest position p such that float(eu[p]) < v, or -1."""
    lo, hi, ans = 0, len(eu) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if float(eu[mid]) < v:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _first_ge(eu: List[str], v: float) -> int:
    """Smallest position p such that v <= float(eu[p]), or len(eu)."""
    lo, hi, ans = 0, len(eu) - 1, len(eu)
    while lo <= hi:
        mid = (lo + hi) // 2
        if v <= float(eu[mid]):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans


# ---------- pretty-printing / debugging --------------------------------------


def icf_human(icf: ICFIndexed, irf: IndexedRandomForest) -> Dict[int, Tuple]:
    """Return a copy with positions replaced by their float thresholds (or
    `None` for ±∞). Useful for asserts and tracing."""
    out: Dict[int, Tuple] = {}
    for i, (b_pos, e_pos) in icf.items():
        eu_i = irf.EU[i]
        b = None if b_pos < 0 else float(eu_i[b_pos])
        e = None if e_pos >= len(eu_i) else float(eu_i[e_pos])
        out[i] = (b, e)
    return out
