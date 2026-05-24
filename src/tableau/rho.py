"""Monotone reason functions ρ over EU-respecting ICFs.

ρ is monotonically non-decreasing in the dominance order
(``ICF ≤ ICF' ⇒ ρ ≤ ρ'``). Two natural choices:

  * ``rho_cells`` — total cell count of the corridor (sum over features of
    the EU-segment length). Strictly monotone; the default.
  * ``rho_constrained_features`` — n_features − (# features fully
    unconstrained). Weakly monotone; useful when interpretability dominates.
"""
from __future__ import annotations

from typing import Sequence

from .cicf import ICFDict


def rho_cells(icf: ICFDict, eu_sizes: Sequence[int]) -> int:
    """Total cell count: ``Σ_f (e_pos − b_pos)`` with the convention that
    ``b_pos = -1`` and ``e_pos = |EU(f)|`` are the infinite sentinels.

    Per feature ``f``, the segment ``(b_pos, e_pos]`` covers ``e_pos − b_pos``
    cells of ``f``. Adding 1 to ``e_pos`` or subtracting 1 from ``b_pos``
    each adds one cell, so this is strictly monotone in the dominance order.
    """
    total = 0
    for f, (b, e) in icf.items():
        total += e - b
    return total


def rho_constrained_features(icf: ICFDict, eu_sizes: Sequence[int]) -> int:
    """``n_features − #fully_unconstrained``. Weakly monotone in dominance
    (extending a feature can turn a constrained feature into an unconstrained
    one but never the reverse)."""
    n_unconstrained = 0
    for f, (b, e) in icf.items():
        if b <= -1 and e >= eu_sizes[f]:
            n_unconstrained += 1
    return len(icf) - n_unconstrained
