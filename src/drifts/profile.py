"""Forest profile under an ICF (docs/majority_check_automata_obdd_milp.md §2.6).

For each tree `T_t`, the *profile* is the set of leaves reachable from some
sample in the corridor:

    Π(T_t, ICF) = { v ∈ V_leaf(T_t) : ∃ x ∈ ICF.  T_t(x) = v }.

Computed by a single recursive descent per tree. At an internal node testing
`(feature_idx=f, threshold_pos=t)` with the corridor's per-feature ICF
positions `(b_pos, e_pos)` on feature `f`:

  - Descend **left** (`x ≤ r`) iff `r > b`, i.e. `t > b_pos`.
    The recursive call inherits `(b_pos, min(e_pos, t))`.
  - Descend **right** (`x > r`) iff `r < e`, i.e. `t < e_pos`.
    The recursive call inherits `(max(b_pos, t), e_pos)`.

Both branches' leaf-sets are unioned. The recursion mutates and restores
`icf` in place to avoid per-call dict copies. Cost: `O(Σ_t |V(T_t)|)`.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .icf import ICFIndexed
from .indexed_forest import IndexedNode, IndexedRandomForest


def per_tree_profile(tree: IndexedNode, icf: ICFIndexed,
                     n_features: int, EU: Dict[int, List[str]]) -> Set[int]:
    """Π(T, ICF): set of `leaf_idx`'s reachable in `tree` from some sample in
    the corridor. Pure positions arithmetic — no float comparisons here."""
    leaves: Set[int] = set()
    # Ensure every feature has an explicit position pair; default = (-1, |EU|).
    local_icf = {i: icf.get(i, (-1, len(EU[i]))) for i in range(n_features)}
    _collect(tree, local_icf, leaves)
    return leaves


def forest_profile(irf: IndexedRandomForest, icf: ICFIndexed) -> List[Set[int]]:
    """Π(RF, ICF) as a list of per-tree reachable-leaf-index sets, in tree-index
    order."""
    return [per_tree_profile(t, icf, irf.n_features, irf.EU) for t in irf.trees]


def _collect(node: IndexedNode, icf: Dict[int, Tuple[int, int]],
             leaves: Set[int]) -> None:
    if node["type"] == "leaf":
        leaves.add(node["leaf_idx"])
        return
    f = node["feature_idx"]
    t = node["threshold_pos"]
    b, e = icf[f]
    # left: t > b
    if t > b:
        new_e = min(e, t)
        icf[f] = (b, new_e)
        _collect(node["low"], icf, leaves)
        icf[f] = (b, e)
    # right: t < e
    if t < e:
        new_b = max(b, t)
        icf[f] = (new_b, e)
        _collect(node["high"], icf, leaves)
        icf[f] = (b, e)
