"""Endpoint Universe (EU) extraction + Convention-B coverage + FSCs.

Self-contained port of `eval-iaxp/our_coverage.py` adapted for this repo.
The EU of feature i is the union of {-inf, +inf} with the set of split
thresholds on feature i across all internal nodes of the trained sklearn RF.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Interval:
    """One (possibly half-) open interval. -inf/+inf for unbounded sides."""

    lo: float = -math.inf
    hi: float = math.inf
    lo_open: bool = True
    hi_open: bool = True

    def contains(self, x: float, *, allow_inf_at_endpoint: bool = False) -> bool:
        if math.isinf(x):
            if not allow_inf_at_endpoint:
                return False
            if x < 0 and self.lo == -math.inf:
                return True
            if x > 0 and self.hi == math.inf:
                return True
            return False
        if x < self.lo or (x == self.lo and self.lo_open):
            return False
        if x > self.hi or (x == self.hi and self.hi_open):
            return False
        return True

    def is_unconstrained(self) -> bool:
        return self.lo == -math.inf and self.hi == math.inf

    def to_list(self) -> List:
        return [
            None if self.lo == -math.inf else self.lo,
            self.lo_open,
            None if self.hi == math.inf else self.hi,
            self.hi_open,
        ]


def extract_eu(rf) -> Dict[int, List[float]]:
    """Per-feature split-threshold universe of a fitted sklearn RandomForestClassifier."""
    n_features = rf.n_features_in_
    eu_sets: Dict[int, set] = {i: set() for i in range(n_features)}
    for est in rf.estimators_:
        t = est.tree_
        for nid in range(t.node_count):
            f = int(t.feature[nid])
            if f < 0:
                continue
            eu_sets[f].add(float(t.threshold[nid]))
    return {i: sorted(eu_sets[i]) for i in range(n_features)}


# ----- parser for RFxpl/GiAXp textual explanations ---------------------------


def parse_explanation_string(expl_text: str, n_features: int) -> Dict[int, Interval]:
    """Parse the literal-by-literal output of GiAXp.py into per-feature intervals."""
    intervals: Dict[int, Interval] = {i: Interval() for i in range(n_features)}
    body = expl_text
    if "IF" in body:
        body = body.split("IF", 1)[1]
    if "THEN" in body:
        body = body.split("THEN", 1)[0]
    for raw in re.split(r"\bAND\b", body):
        conj = raw.strip().strip("()[]")
        first = re.split(r"\bOR\b", conj)[0].strip().strip("()[]")
        m = _parse_one_literal(first)
        if m is None:
            continue
        idx, iv = m
        if 0 <= idx < n_features:
            intervals[idx] = iv
    return intervals


def _parse_one_literal(lit: str) -> Tuple[int, Interval] | None:
    s = lit.replace(" ", "")
    m = re.fullmatch(
        r"(?P<lo>-?\d+(?:\.\d+)?)(?P<lop><=|<)f(?P<idx>\d+)(?P<hop><=|<)(?P<hi>-?\d+(?:\.\d+)?)",
        s,
    )
    if m:
        return (
            int(m["idx"]),
            Interval(
                lo=float(m["lo"]),
                hi=float(m["hi"]),
                lo_open=(m["lop"] == "<"),
                hi_open=(m["hop"] == "<"),
            ),
        )
    m = re.fullmatch(r"f(?P<idx>\d+)(?P<op>>=|>)(?P<v>-?\d+(?:\.\d+)?)", s)
    if m:
        return int(m["idx"]), Interval(lo=float(m["v"]), lo_open=(m["op"] == ">"))
    m = re.fullmatch(r"f(?P<idx>\d+)(?P<op><=|<)(?P<v>-?\d+(?:\.\d+)?)", s)
    if m:
        return int(m["idx"]), Interval(hi=float(m["v"]), hi_open=(m["op"] == "<"))
    return None


# ----- coverage and FSCs -----------------------------------------------------


def coverage_convB(intervals: Dict[int, Interval], eu: Dict[int, List[float]]) -> float:
    """Convention-B coverage: average over features of (# EU points in σ) / |EU|.

    EU here is augmented with {-inf, +inf}; +/-inf counts as inside iff the
    interval is unbounded on that side.
    """
    fracs = []
    for i, iv in intervals.items():
        pts = list(eu.get(i, [])) + [-math.inf, math.inf]
        denom = len(pts)
        num = sum(1 for e in pts if iv.contains(e, allow_inf_at_endpoint=True))
        fracs.append(num / denom if denom else 0.0)
    return sum(fracs) / len(fracs) if fracs else 0.0


def fsc_s(intervals: Dict[int, Interval], eu: Dict[int, List[float]]) -> float:
    """IJCAI 2025 size measure: sum over constrained features of log s_i(E_i),
    with s_i(E_i) the count of finite EU thresholds of feature i lying inside E_i.
    Unconstrained features contribute 0.
    """
    total = 0.0
    for i, iv in intervals.items():
        if iv.is_unconstrained():
            continue
        finite_pts = eu.get(i, [])
        s_i = sum(1 for e in finite_pts if iv.contains(e))
        if s_i > 0:
            total += math.log(s_i)
    return total


def constrained_feature_ids(intervals: Dict[int, Interval]) -> List[int]:
    return sorted(i for i, iv in intervals.items() if not iv.is_unconstrained())
