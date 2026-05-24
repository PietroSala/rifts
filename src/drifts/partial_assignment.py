"""Partial leaf assignment α (docs/majority_check_automata_obdd_milp.md §2.7).

Per-tree map `α_t: leaf_idx → {0, 1, ⊥}`. We only store the *decided* entries
(0 or 1); leaves absent from the map are implicitly ⊥. JSON-friendly.

Canonical form (per tree `t` with `n_t` leaves):

  * All-⊥        — tree absent from `decided` (no constraint on `t`).
  * Single-1     — exactly one leaf at 1; the other leaves are implicitly 0
                   (canonical dominance) and are NOT stored.
  * Multi-0      — some leaves at 0, none at 1, and at least two leaves
                   still ⊥.
  * Extinguished — every leaf at 0 (corridor-empty signal); reachable only
                   via bulk construction, never via incremental `set()`.

Set-level invariants enforced by `set()`:

  * ≤ 1 one-mark per tree (contradiction raises).
  * `n_t − 1` 0-marks with one ⊥ leaf → promote that leaf to 1, drop the 0s.
  * Writing 1 on a leaf already at 1 (same v) is idempotent; writing the
    opposite value contradicts and raises.

The specialization order `α' ⊐ α` (α' refines α) is set-containment of decided
entries with canonical dominance: a tree-α with a single 1-mark at v⋆ implicitly
forbids every other leaf, so it dominates any tree-α° whose 0-marks do not
include v⋆.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence


@dataclass(frozen=False)
class PartialAssignment:
    """Sparse partial assignment in canonical form."""
    decided: Dict[int, Dict[int, int]] = field(default_factory=dict)
    n_leaves_per_tree: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.n_leaves_per_tree = tuple(self.n_leaves_per_tree)
        for t in list(self.decided.keys()):
            self._canonicalise_tree(t)

    # ----- canonicalisation -------------------------------------------------

    def _canonicalise_tree(self, t: int) -> None:
        m = self.decided.get(t)
        if not m:
            self.decided.pop(t, None)
            return
        ones = [v for v, val in m.items() if val == 1]
        if len(ones) > 1:
            raise ValueError(
                f"tree {t}: multiple 1-marks at leaves {sorted(ones)}"
            )
        if ones:
            self.decided[t] = {ones[0]: 1}
            return
        # only 0-marks
        if t < len(self.n_leaves_per_tree):
            n_t = self.n_leaves_per_tree[t]
            zeros = list(m.keys())
            if len(zeros) == n_t - 1:
                forced = next(v for v in range(n_t) if v not in m)
                self.decided[t] = {forced: 1}
            # len == n_t  →  extinguished, leave as-is
            # len <  n_t-1 →  multi-0, leave as-is

    # ----- basic accessors --------------------------------------------------

    def value(self, tree_idx: int, leaf_idx: int) -> int | None:
        return self.decided.get(tree_idx, {}).get(leaf_idx)

    def set(self, tree_idx: int, leaf_idx: int, v: int) -> None:
        if v not in (0, 1):
            raise ValueError(f"value must be 0 or 1, got {v!r}")
        m = self.decided.setdefault(tree_idx, {})
        existing = m.get(leaf_idx)
        if existing is not None and existing != v:
            raise ValueError(
                f"contradiction on (tree={tree_idx}, leaf={leaf_idx}): "
                f"existing={existing}, new={v}"
            )
        m[leaf_idx] = v
        self._canonicalise_tree(tree_idx)

    def n_decided(self) -> int:
        return sum(len(v) for v in self.decided.values())

    def items(self) -> Iterable[tuple[int, int, int]]:
        for t, m in self.decided.items():
            for v, val in m.items():
                yield t, v, val

    def canonical(self) -> tuple:
        """Hashable, sortable canonical tuple of (tree, leaf, value) triples."""
        return tuple(sorted(self.items()))

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, PartialAssignment)
                and self.canonical() == other.canonical())

    # ----- specialization order --------------------------------------------

    def specializes(self, other: "PartialAssignment") -> bool:
        """`self ⊐ other` under canonical dominance.

        A tree-α with a single 1-mark at v⋆ implicitly forbids every other
        leaf in that tree; hence it dominates any tree-α° whose 0-marks do
        not include v⋆.
        """
        for t, m_other in other.decided.items():
            m_self = self.decided.get(t, {})
            ones_other = [v for v, val in m_other.items() if val == 1]
            ones_self = [v for v, val in m_self.items() if val == 1]

            if ones_other:
                v_circ = ones_other[0]
                # self must force the same v⋆
                if v_circ not in ones_self:
                    return False
                # check any explicit 0-marks in other (canonical form should
                # not have them, but guard against non-canonical inputs)
                for v, val in m_other.items():
                    if val == 0 and v == v_circ:
                        return False
            else:
                # other carries only 0-marks
                if ones_self:
                    v_star = ones_self[0]
                    # self forces v⋆ = 1 → implicitly forbids every v ≠ v⋆
                    if m_other.get(v_star) == 0:
                        return False
                    # other's 0-marks on v ≠ v⋆ are absorbed by canonical dominance
                else:
                    # both have only 0-marks; require explicit set-containment
                    for v, val in m_other.items():
                        if m_self.get(v) != val:
                            return False
        return True

    # ----- consistency ------------------------------------------------------

    def is_consistent(self, n_leaves_per_tree: Sequence[int] | None = None) -> bool:
        """Per-tree: ≤ 1 one-mark AND not every leaf 0 (corridor non-empty).

        If `n_leaves_per_tree` is None, falls back to the instance attribute.
        """
        nlpt = n_leaves_per_tree if n_leaves_per_tree is not None else self.n_leaves_per_tree
        for t, n_t in enumerate(nlpt):
            m = self.decided.get(t, {})
            ones = sum(1 for val in m.values() if val == 1)
            if ones > 1:
                return False
            zeros = sum(1 for val in m.values() if val == 0)
            if zeros == n_t:
                return False
        return True

    def is_complete(self, n_leaves_per_tree: Sequence[int] | None = None) -> bool:
        """Every tree determined (canonical: each has exactly one 1-mark)."""
        nlpt = n_leaves_per_tree if n_leaves_per_tree is not None else self.n_leaves_per_tree
        for t in range(len(nlpt)):
            m = self.decided.get(t, {})
            ones = sum(1 for val in m.values() if val == 1)
            if ones != 1:
                return False
        return True

    # ----- B-cache full-assignment admissibility (legacy) -------------------

    def admits(self, full: "PartialAssignment") -> bool:
        """Whether a fully-decided `full` is still admissible under `self`.

        Legacy helper; with the v1 cache design B uses `specializes`, not
        `admits`. Kept for backwards compatibility with older cache code.
        """
        for t, m in full.decided.items():
            for v, val in m.items():
                if val == 1 and self.value(t, v) == 0:
                    return False
        return True

    # ----- serialisation ----------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "decided": {
                str(t): {str(v): val for v, val in sorted(m.items())}
                for t, m in sorted(self.decided.items())
            },
            "n_leaves_per_tree": list(self.n_leaves_per_tree),
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, s: str,
                  n_leaves_per_tree: Sequence[int] | None = None) -> "PartialAssignment":
        raw = json.loads(s)
        if isinstance(raw, dict) and "decided" in raw and "n_leaves_per_tree" in raw:
            decided = {int(t): {int(v): int(val) for v, val in m.items()}
                       for t, m in raw["decided"].items()}
            nlpt = tuple(raw["n_leaves_per_tree"])
        else:
            # legacy format: flat decided map at the top level
            decided = {int(t): {int(v): int(val) for v, val in m.items()}
                       for t, m in raw.items()}
            nlpt = tuple(n_leaves_per_tree or ())
        return cls(decided=decided, n_leaves_per_tree=nlpt)

    @classmethod
    def initial_from_profile(cls, profile: Sequence[Sequence[int]],
                             n_leaves_per_tree: Sequence[int]) -> "PartialAssignment":
        """Build α from a forest profile.

        For each tree `t`, set α_t(v) = 0 for v ∉ profile[t]. Canonicalisation
        then collapses any tree whose profile is a singleton to a single
        1-mark on the reached leaf, and any tree whose profile is empty stays
        in the extinguished state.
        """
        decided: Dict[int, Dict[int, int]] = {}
        for t, reachable in enumerate(profile):
            reachable_set = set(reachable)
            forbidden = {v: 0 for v in range(n_leaves_per_tree[t])
                         if v not in reachable_set}
            if forbidden:
                decided[t] = forbidden
        return cls(decided=decided, n_leaves_per_tree=tuple(n_leaves_per_tree))
