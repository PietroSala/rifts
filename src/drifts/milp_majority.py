"""MILP majority oracle, closed-form per §4.2 of the spec doc.

The MILP works at the **per-(tree, label) level** — never per leaf, never per
cell. Given the per-tree profile `Π(T_t, ICF)`, the current partial α and an
optional derived δ, the available-leaf and available-label sets per tree are

    A_t = { v ∈ Π_t : α_t(v) ≠ 0  AND  δ_t(v) ≠ 0 }
    L_t = { lab(v) : v ∈ A_t }.

For each non-seed competitor `c ≠ c⋆`, §4.2 gives the closed form

    Adv(c) = Σ_{determined trees} ( 1[ℓ_t = c] − 1[ℓ_t = c⋆] )
           + |{ free trees t : c ∈ L_t }|

where a tree is *determined* if |L_t| = 1 (vote forced to `ℓ_t`) and *free*
otherwise. This is a label histogram across trees — no solver needed.

The verdict accounts for our Def-3 lex-first tie-break (label-index order
coincides with lex order by construction in `sklearn_io.from_sklearn`):

  - `c⋆` wins under lex-first iff for every `c ≠ c⋆`:
      * if `c > c⋆`  (lex):  Adv(c) ≤ 0;
      * if `c < c⋆`  (lex):  Adv(c) < 0.

  Negation gives the "c⋆ could lose" criterion the relaxation tests.

`Bad` is declared only at a *slackless* state (every L_t is a singleton) where
the relaxation coincides with the real vote. Otherwise: `good` (`c⋆` certified
under relaxation) or `inconclusive` (keep learning).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .indexed_forest import IndexedRandomForest
from .partial_assignment import PartialAssignment


Verdict = str  # "good" | "bad" | "inconclusive"


@dataclass
class MILPResult:
    verdict: Verdict
    adv: Dict[int, int]           # competitor label_idx → Adv(c)
    slackless: bool               # every tree's L_t is a singleton
    determined_labels: List[Optional[int]]  # per tree: forced label or None if free
    L_per_tree: List[Set[int]]    # per tree: available label set
    witness: Optional[PartialAssignment] = None  # set only when verdict == "bad"

    def __repr__(self) -> str:
        return (f"MILPResult(verdict={self.verdict!r}, "
                f"adv={self.adv}, slackless={self.slackless})")


def majority_check(
    irf: IndexedRandomForest,
    profile: Sequence[Set[int]],
    alpha: PartialAssignment,
    c_star: int,
    delta: Optional[PartialAssignment] = None,
) -> MILPResult:
    """Compute Adv(c) for every competitor and return the verdict.

    Arguments:
      irf:      the indexed forest (provides label-of-leaf via leaves_per_tree()).
      profile:  per-tree set of reachable leaf_idx's (Π(T_t, ICF)).
      alpha:    current partial assignment (committed leaves).
      c_star:   seed-class label index.
      delta:    optional derived assignment (forced by the OBDD). When `None`,
                treated as everywhere-undecided.
    """
    n_labels = irf.n_labels
    leaves_per_tree = irf.leaves_per_tree()

    # ---- Per-tree L_t (available label set) ---------------------------------
    L_per_tree: List[Set[int]] = []
    for t, prof_t in enumerate(profile):
        labels: Set[int] = set()
        for leaf_idx in prof_t:
            if alpha.value(t, leaf_idx) == 0:
                continue
            if delta is not None and delta.value(t, leaf_idx) == 0:
                continue
            labels.add(leaves_per_tree[t][leaf_idx]["label_idx"])
        L_per_tree.append(labels)

    # ---- Determined / free split + histograms -------------------------------
    K = [0] * n_labels                # determined-tree contributions
    free_count = [0] * n_labels       # free trees with c in L_t
    determined_labels: List[Optional[int]] = []
    slackless = True
    for L_t in L_per_tree:
        if len(L_t) == 0:
            # Tree extinguished — every leaf forbidden. Either α is faulty
            # (caller should have caught this) or every leaf in the profile was
            # forbidden by α/δ. We treat the tree as contributing nothing to
            # either side and leave `determined_labels[t] = None`; the verifier
            # is responsible for declaring the state faulty.
            determined_labels.append(None)
            slackless = False
            continue
        if len(L_t) == 1:
            (ell_t,) = L_t
            K[ell_t] += 1
            determined_labels.append(ell_t)
        else:
            slackless = False
            determined_labels.append(None)
            for c in L_t:
                free_count[c] += 1

    # ---- Adv(c) closed form (§4.2) -----------------------------------------
    adv: Dict[int, int] = {}
    for c in range(n_labels):
        if c == c_star:
            continue
        adv[c] = (K[c] - K[c_star]) + free_count[c]

    # ---- Verdict under Def-3 lex-first tie-break ---------------------------
    bad_threats: List[int] = []
    for c, a in adv.items():
        # c can defeat c* iff:
        #   c > c_star (lex) and a > 0   (strict win required)
        #   c < c_star (lex) and a >= 0  (tie suffices since c is lex-smaller)
        if c > c_star and a > 0:
            bad_threats.append(c)
        elif c < c_star and a >= 0:
            bad_threats.append(c)

    if not bad_threats:
        return MILPResult(
            verdict="good", adv=adv, slackless=slackless,
            determined_labels=determined_labels, L_per_tree=L_per_tree,
        )

    # Some threat exists. Bad is only definitive at slackless states (§4.4).
    if not slackless:
        return MILPResult(
            verdict="inconclusive", adv=adv, slackless=False,
            determined_labels=determined_labels, L_per_tree=L_per_tree,
        )

    # Slackless + threat → bad with the determined-label witness.
    witness = _build_witness(determined_labels, profile, alpha, delta,
                             leaves_per_tree, irf)
    return MILPResult(
        verdict="bad", adv=adv, slackless=True,
        determined_labels=determined_labels, L_per_tree=L_per_tree,
        witness=witness,
    )


def majority_check_min(
    irf: IndexedRandomForest,
    profile: Sequence[Set[int]],
    alpha: PartialAssignment,
    c_star: int,
    delta: Optional[PartialAssignment] = None,
) -> MILPResult:
    """Adv_min(c) closed form — Bad detection on partial α (§4.2 dual).

    Per free tree, the worst-case contribution to ``Adv(c) = votes(c) − votes(c⋆)``
    is

      * ``-1`` if ``c⋆ ∈ L_t``  (the tree can be made to vote ``c⋆``);
      * ``0``  otherwise           (the tree can be made to vote some label ≠ c).

    Note the second contribution does not depend on ``c`` — even if ``c ∈ L_t``
    the free tree (|L_t| ≥ 2) can pick some label ≠ c. Hence

        Adv_min(c) = (K[c] − K[c⋆]) − |{ free trees t : c⋆ ∈ L_t }|

    where ``K[c]`` is the count of determined trees with forced label ``c``.

    Bad verdict under Def-3 lex-first iff some competitor ``c`` satisfies:

      * ``c > c⋆`` (lex):  ``Adv_min(c) > 0``  (strict win even in worst case);
      * ``c < c⋆`` (lex):  ``Adv_min(c) ≥ 0``  (tie suffices for the lex-smaller).

    Returns ``MILPResult(verdict ∈ {"bad", "inconclusive"})`` with ``witness=None``;
    the verifier stores ``α`` (smaller) into B without needing a full witness.
    """
    n_labels = irf.n_labels
    leaves_per_tree = irf.leaves_per_tree()

    L_per_tree: List[Set[int]] = []
    for t, prof_t in enumerate(profile):
        labels: Set[int] = set()
        for leaf_idx in prof_t:
            if alpha.value(t, leaf_idx) == 0:
                continue
            if delta is not None and delta.value(t, leaf_idx) == 0:
                continue
            labels.add(leaves_per_tree[t][leaf_idx]["label_idx"])
        L_per_tree.append(labels)

    K = [0] * n_labels
    free_with_cstar = 0
    determined_labels: List[Optional[int]] = []
    slackless = True
    for L_t in L_per_tree:
        if len(L_t) == 0:
            determined_labels.append(None)
            slackless = False
            continue
        if len(L_t) == 1:
            (ell_t,) = L_t
            K[ell_t] += 1
            determined_labels.append(ell_t)
        else:
            slackless = False
            determined_labels.append(None)
            if c_star in L_t:
                free_with_cstar += 1

    adv: Dict[int, int] = {}
    for c in range(n_labels):
        if c == c_star:
            continue
        adv[c] = (K[c] - K[c_star]) - free_with_cstar

    bad_threats: List[int] = []
    for c, a in adv.items():
        if c > c_star and a > 0:
            bad_threats.append(c)
        elif c < c_star and a >= 0:
            bad_threats.append(c)

    verdict = "bad" if bad_threats else "inconclusive"
    return MILPResult(
        verdict=verdict, adv=adv, slackless=slackless,
        determined_labels=determined_labels, L_per_tree=L_per_tree,
        witness=None,
    )


def _build_witness(
    determined_labels: List[Optional[int]],
    profile: Sequence[Set[int]],
    alpha: PartialAssignment,
    delta: Optional[PartialAssignment],
    leaves_per_tree,
    irf: IndexedRandomForest,
) -> PartialAssignment:
    """Construct a complete per-tree leaf-tuple α^× consistent with the slackless
    state: per tree, pick any available leaf whose label is the forced ℓ_t,
    mark it 1, mark every other leaf of the tree 0.
    """
    full = PartialAssignment()
    for t, ell_t in enumerate(determined_labels):
        if ell_t is None:
            # Should not happen at slackless; fall back to leaving the tree
            # blank (the verifier will reject the witness as ill-formed).
            continue
        # Find one leaf in profile[t] with label ell_t and not forbidden by α/δ.
        chosen: Optional[int] = None
        for leaf_idx in profile[t]:
            if alpha.value(t, leaf_idx) == 0:
                continue
            if delta is not None and delta.value(t, leaf_idx) == 0:
                continue
            if leaves_per_tree[t][leaf_idx]["label_idx"] == ell_t:
                chosen = leaf_idx
                break
        if chosen is None:
            continue
        # mark every leaf of tree t as 0 except `chosen` which is 1
        for leaf_idx in range(irf.n_leaves_per_tree()[t]):
            full.set(t, leaf_idx, 1 if leaf_idx == chosen else 0)
    return full
