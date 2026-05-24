"""Constrained ICF triple (inner, floor, ceiling) — the tableau node payload.

Each ICF is stored in EU-position form: a ``dict[int, tuple[int, int]]``
mapping feature_idx → (b_pos, e_pos). The sentinels follow
``drifts.icf.ICFIndexed``:

  * ``b_pos = -1``   ↔  ``b = -∞``
  * ``e_pos = |EU|`` ↔  ``e = +∞``

A cICF must satisfy ``ICF↓ ≤ ICF ≤ ICF↑`` in the dominance order (per
feature ``b↑ ≤ b ≤ b↓`` and ``e↓ ≤ e ≤ e↑``).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


ICFDict = Dict[int, Tuple[int, int]]


# ---------- canonical serialisation -----------------------------------------


def canonical_icf_key(icf: ICFDict) -> str:
    """Stable JSON encoding (sorted by feature_idx)."""
    return json.dumps(
        {str(i): [b, e] for i, (b, e) in sorted(icf.items())},
        separators=(",", ":"),
    )


def canonical_node_id(inner: ICFDict, floor: ICFDict, ceiling: ICFDict,
                      class_label: int) -> str:
    """Stable SHA-1 of the triple + class. Used as the tableau node id."""
    h = hashlib.sha1()
    h.update(canonical_icf_key(inner).encode())
    h.update(b"|")
    h.update(canonical_icf_key(floor).encode())
    h.update(b"|")
    h.update(canonical_icf_key(ceiling).encode())
    h.update(b"|c=")
    h.update(str(class_label).encode())
    return h.hexdigest()


# ---------- the cICF dataclass ----------------------------------------------


@dataclass(frozen=True)
class CICF:
    """A constrained ICF triple ``(inner, floor, ceiling)`` with the class
    label carried from the root.

    The data is immutable; transformations produce a fresh ``CICF``.
    """
    inner: ICFDict
    floor: ICFDict
    ceiling: ICFDict
    class_label: int

    # ----- node identity ----------------------------------------------------

    def node_id(self) -> str:
        return canonical_node_id(self.inner, self.floor, self.ceiling,
                                 self.class_label)

    # ----- saturation -------------------------------------------------------

    def is_expansion_saturated(self) -> bool:
        return self.inner == self.ceiling

    def is_shrink_saturated(self) -> bool:
        return self.inner == self.floor

    # ----- serialisation ----------------------------------------------------

    def to_payload(self) -> Dict[str, str]:
        """The dict that gets stored at ``tableau:node:<id>`` (all values
        are strings so it round-trips through ``HMSET`` / ``HGETALL``)."""
        return {
            "id": self.node_id(),
            "icf": canonical_icf_key(self.inner),
            "icf_lo": canonical_icf_key(self.floor),
            "icf_hi": canonical_icf_key(self.ceiling),
            "class": str(self.class_label),
        }

    @classmethod
    def from_payload(cls, payload: Dict) -> "CICF":
        def _parse(s: str) -> ICFDict:
            raw = json.loads(s)
            return {int(k): tuple(v) for k, v in raw.items()}
        return cls(
            inner=_parse(payload["icf"]),
            floor=_parse(payload["icf_lo"]),
            ceiling=_parse(payload["icf_hi"]),
            class_label=int(payload["class"]),
        )


# ---------- saturation helpers (function form) -------------------------------


def is_expansion_saturated(c: CICF) -> bool:
    return c.is_expansion_saturated()


# ---------- admissibility under the ceiling ---------------------------------


def can_extend_left(icf: ICFDict, ceiling: ICFDict, feat: int) -> bool:
    """True iff ``ICF ⊕ ⟵f`` is admissible (the new left endpoint is ≥
    the ceiling's left endpoint, i.e. ``b - 1 ≥ b_ceiling``).
    """
    b, _e = icf[feat]
    b_hi, _ = ceiling[feat]
    return b > -1 and (b - 1) >= b_hi


def can_extend_right(icf: ICFDict, ceiling: ICFDict, feat: int,
                     eu_size: int) -> bool:
    """True iff ``ICF ⊕ f⟶`` is admissible."""
    _b, e = icf[feat]
    _, e_hi = ceiling[feat]
    return e < eu_size and (e + 1) <= e_hi


# ---------- Next (one-step expansion on every admissible side) ---------------


def next_icf(c: CICF, eu_sizes: Sequence[int]) -> CICF:
    """Apply one EU step outward on every admissible side of every feature.

    Blocked directions (already at the ceiling) are left unchanged. The floor
    and ceiling are preserved. If no extension is admissible the input is
    returned unchanged (already expansion-saturated).
    """
    new_inner: ICFDict = {}
    changed = False
    for f, (b, e) in c.inner.items():
        n = eu_sizes[f]
        b_hi, e_hi = c.ceiling[f]
        new_b = b - 1 if (b > -1 and (b - 1) >= b_hi) else b
        new_e = e + 1 if (e < n and (e + 1) <= e_hi) else e
        if new_b != b or new_e != e:
            changed = True
        new_inner[f] = (new_b, new_e)
    if not changed:
        return c
    return CICF(inner=new_inner, floor=c.floor, ceiling=c.ceiling,
                class_label=c.class_label)


# ---------- bipartite shrinkage ---------------------------------------------


def _shrink_right(c: CICF, f: int) -> Optional[Tuple[CICF, CICF]]:
    """Return (child_A, child_B) for a right-direction bipartite shrinkage on
    feature ``f``, or ``None`` if neither side is admissible."""
    b, e = c.inner[f]
    b_lo, e_lo = c.floor[f]
    b_hi, e_hi = c.ceiling[f]
    e_circ = e - 1                          # max EU position < e (≥ -1 sentinel)

    a_admissible = (b < e_circ) and (e_circ >= e_lo)
    b_admissible = (e_lo < e_circ <= e)

    if not a_admissible and not b_admissible:
        return None

    children: List[CICF] = []
    if a_admissible:
        inner_a = dict(c.inner);   inner_a[f]   = (b, e_circ)
        ceil_a  = dict(c.ceiling); ceil_a[f]    = (b_hi, e_circ)
        children.append(CICF(inner=inner_a, floor=c.floor, ceiling=ceil_a,
                             class_label=c.class_label))
    if b_admissible:
        floor_b = dict(c.floor);   floor_b[f]   = (b_lo, e_circ)
        children.append(CICF(inner=c.inner, floor=floor_b, ceiling=c.ceiling,
                             class_label=c.class_label))
    # The B&B contract expects two children when a shrinkage is "admissible".
    # If only one side fits, the caller must handle it (treat as a single
    # advance with no sibling; the search remains complete).
    if len(children) == 1:
        return (children[0], children[0])
    return (children[0], children[1])


def _shrink_left(c: CICF, f: int) -> Optional[Tuple[CICF, CICF]]:
    b, e = c.inner[f]
    b_lo, e_lo = c.floor[f]
    b_hi, e_hi = c.ceiling[f]
    b_circ = b + 1                          # min EU position > b

    a_admissible = (b_circ < e) and (b_circ <= b_lo)
    b_admissible = (b <= b_circ < b_lo)

    if not a_admissible and not b_admissible:
        return None

    children: List[CICF] = []
    if a_admissible:
        inner_a = dict(c.inner);   inner_a[f]   = (b_circ, e)
        ceil_a  = dict(c.ceiling); ceil_a[f]    = (b_circ, e_hi)
        children.append(CICF(inner=inner_a, floor=c.floor, ceiling=ceil_a,
                             class_label=c.class_label))
    if b_admissible:
        floor_b = dict(c.floor);   floor_b[f]   = (b_circ, e_lo)
        children.append(CICF(inner=c.inner, floor=floor_b, ceiling=c.ceiling,
                             class_label=c.class_label))
    if len(children) == 1:
        return (children[0], children[0])
    return (children[0], children[1])


def bipartite_shrinkage(c: CICF, eu_sizes: Sequence[int],
                        rng: random.Random | None = None
                        ) -> Optional[Tuple[CICF, CICF, int, str]]:
    """Pick a random admissible shrinkage ``(f, d)`` and return
    ``(child_A, child_B, f, d)`` with ``d ∈ {"left", "right"}``. Returns
    ``None`` if no shrinkage is admissible on any feature/direction.
    """
    rng = rng or random.Random()
    feats = list(c.inner.keys())
    rng.shuffle(feats)
    dirs = ["left", "right"]
    for f in feats:
        rng.shuffle(dirs)
        for d in dirs:
            res = _shrink_right(c, f) if d == "right" else _shrink_left(c, f)
            if res is not None:
                return (res[0], res[1], f, d)
    return None
