"""Convert a fitted `sklearn.ensemble.RandomForestClassifier` to an
`IndexedRandomForest` (Definition 8 of `icf-foundations.md`).

The sklearn forest carries unnamed features (just positional integers); we
generate **zero-padded** feature names `f000…fNNN` so that the natural
lexicographic order coincides with the integer index order — keeping
`phi_F` canonical per Definition 8.1.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .indexed_forest import IndexedNode, IndexedRandomForest


def from_sklearn(rf, dataset: str = "") -> IndexedRandomForest:
    """Build an `IndexedRandomForest` from a fitted sklearn RF.

    All numeric thresholds are stored as `repr(float(x))` strings, so a round
    trip through Redis returns bit-exact numbers when cast back via
    `float(...)`.
    """
    n_features = int(rf.n_features_in_)
    n_digits = max(1, int(math.ceil(math.log10(max(n_features, 1) + 1))))
    feature_names = [f"f{i:0{n_digits}d}" for i in range(n_features)]
    phi_F: Dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    classes_sorted = sorted(str(c) for c in rf.classes_)
    phi_L: Dict[str, int] = {c: i for i, c in enumerate(classes_sorted)}
    class_to_label_idx: Dict[int, int] = {
        sklearn_pos: phi_L[str(c)] for sklearn_pos, c in enumerate(rf.classes_)
    }

    # ---- EU: per-feature sorted threshold sequence (Def 8.2) ----
    eu_sets: Dict[int, set] = {i: set() for i in range(n_features)}
    for est in rf.estimators_:
        t = est.tree_
        for nid in range(t.node_count):
            f = int(t.feature[nid])
            if f < 0:
                continue
            eu_sets[f].add(repr(float(t.threshold[nid])))
    EU: Dict[int, List[str]] = {
        i: sorted(eu_sets[i], key=lambda s: float(s)) for i in range(n_features)
    }
    threshold_pos: Dict[int, Dict[str, int]] = {
        i: {s: p for p, s in enumerate(EU[i])} for i in range(n_features)
    }

    # ---- per-tree recursive indexed structure (Def 7) ----
    trees: List[IndexedNode] = []
    for est in rf.estimators_:
        trees.append(_convert_tree(est.tree_, threshold_pos, class_to_label_idx))

    return IndexedRandomForest(
        dataset=dataset,
        phi_F=phi_F,
        phi_L=phi_L,
        EU=EU,
        trees=trees,
    )


def _convert_tree(t, threshold_pos, class_to_label_idx) -> IndexedNode:
    leaf_counter = [0]
    return _convert_node(t, 0, threshold_pos, class_to_label_idx, leaf_counter)


def _convert_node(t, node_id: int, threshold_pos, class_to_label_idx,
                  leaf_counter) -> IndexedNode:
    feat = int(t.feature[node_id])
    if feat < 0:  # leaf
        values = t.value[node_id]
        if values.ndim == 2:
            values = values[0]
        sklearn_pos = int(np.argmax(values))
        leaf_idx = leaf_counter[0]
        leaf_counter[0] += 1
        return {
            "type": "leaf",
            "label_idx": class_to_label_idx[sklearn_pos],
            "leaf_idx": leaf_idx,
        }
    repr_thr = repr(float(t.threshold[node_id]))
    return {
        "type": "internal",
        "feature_idx": feat,
        "threshold_pos": threshold_pos[feat][repr_thr],
        "low":  _convert_node(t, int(t.children_left[node_id]),  threshold_pos,
                              class_to_label_idx, leaf_counter),
        "high": _convert_node(t, int(t.children_right[node_id]), threshold_pos,
                              class_to_label_idx, leaf_counter),
    }
