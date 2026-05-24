"""Indexed Random Forest — Definitions 7 and 8 of `icf-foundations.md`.

The forest carries:
  - `phi_F`: feature-name → feature-index (lexicographic by *padded* name).
  - `phi_L`: label-name → label-index (lexicographic).
  - `EU`: dict[feature_idx] → strictly increasing list of thresholds; each
    threshold stored as a string (lossless `repr(float)`) so that the
    round-trip through Redis is bit-exact. Cast to float on the fly when
    arithmetic is needed.
  - `trees`: list of `IndexedTree`s (recursive dicts). Leaves are
    `{"type": "leaf", "label_idx": k}`; internal nodes are
    `{"type": "internal", "feature_idx": i, "threshold_pos": t,
      "low": <subtree>, "high": <subtree>}`.

Both `feature_idx` and `threshold_pos` are integers ranging over
`{0, …, n_features-1}` and `{0, …, |EU(i)|-1}` respectively. The original
threshold value is recoverable via `float(EU[i][threshold_pos])`. The original
feature name and label are recoverable by inverting `phi_F` / `phi_L`.

This module deals only with the in-memory dataclass and a single
`predict()` helper; sklearn import lives in `sklearn_io.py` and the Redis
persistence lives in `cache/store.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


IndexedNode = Dict[str, Any]  # recursive; see module docstring


@dataclass
class IndexedRandomForest:
    """In-memory mirror of the Redis-persisted indexed forest."""

    dataset: str
    phi_F: Dict[str, int]                 # feature name -> index
    phi_L: Dict[str, int]                 # label name -> index
    EU: Dict[int, List[str]]              # feature_idx -> sorted [repr(float), ...]
    trees: List[IndexedNode] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.phi_F)

    @property
    def n_labels(self) -> int:
        return len(self.phi_L)

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    def n_leaves_per_tree(self) -> List[int]:
        """Number of leaves per indexed tree, in tree-index order."""
        return [_count_leaves(t) for t in self.trees]

    def leaves_per_tree(self) -> List[List[IndexedNode]]:
        """For each tree, the list of leaf nodes in `leaf_idx` order."""
        out: List[List[IndexedNode]] = []
        for tree in self.trees:
            leaves: List[IndexedNode] = []
            _collect_leaves(tree, leaves)
            leaves.sort(key=lambda n: n["leaf_idx"])
            out.append(leaves)
        return out

    def eu_floats(self, feature_idx: int) -> List[float]:
        """Convenience: per-feature EU as floats (cast on the fly)."""
        return [float(s) for s in self.EU[feature_idx]]

    def predict(self, x: List[float]) -> int:
        """Majority-vote prediction on a raw sample vector (feature_idx → value).

        `x[i]` must be the value of the feature with feature-index `i` (so the
        caller is responsible for arranging the sample in φ_F order). Returns
        the **label index**; the human-readable label is recoverable via
        `phi_L_inv = {v: k for k, v in phi_L.items()}`.
        """
        if len(x) != self.n_features:
            raise ValueError(
                f"sample has {len(x)} values but forest has {self.n_features} features"
            )
        votes: Dict[int, int] = {}
        for tree in self.trees:
            label_idx = _predict_tree(tree, x, self.EU)
            votes[label_idx] = votes.get(label_idx, 0) + 1
        return max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def per_tree_labels_from_icf(self, icf: Dict[int, tuple]) -> List[int]:
        """Walk every tree using ICF positions; return one `label_idx` per tree.

        At an internal node with `(feature_idx, threshold_pos)`, the test
        `x[feature_idx] ≤ EU[feature_idx][threshold_pos]` is decided from the
        ICF interval `(b_pos, e_pos]`:
          - `threshold_pos ≥ e_pos`  ⇒ `x ≤ thr` (go LEFT).
          - `threshold_pos ≤ b_pos`  ⇒ `x > thr` (go RIGHT).
          - otherwise the ICF straddles the threshold and the answer is
            ambiguous — only happens for non-trivial ICFs and is reported as
            an error here. Step 2+ will branch on it during enumeration.
        """
        labels: List[int] = []
        for tree in self.trees:
            labels.append(_predict_tree_from_icf(tree, icf))
        return labels


def _predict_tree_from_icf(node: IndexedNode, icf: Dict[int, tuple]) -> int:
    while node["type"] == "internal":
        feat = node["feature_idx"]
        thr_pos = node["threshold_pos"]
        b_pos, e_pos = icf[feat]
        if thr_pos >= e_pos:
            node = node["low"]
        elif thr_pos <= b_pos:
            node = node["high"]
        else:
            raise ValueError(
                f"ICF straddles threshold at feature {feat}: "
                f"b_pos={b_pos}, e_pos={e_pos}, thr_pos={thr_pos}"
            )
    return node["label_idx"]


def _count_leaves(node: IndexedNode) -> int:
    if node["type"] == "leaf":
        return 1
    return _count_leaves(node["low"]) + _count_leaves(node["high"])


def _collect_leaves(node: IndexedNode, out: List[IndexedNode]) -> None:
    if node["type"] == "leaf":
        out.append(node)
        return
    _collect_leaves(node["low"], out)
    _collect_leaves(node["high"], out)


def _predict_tree(node: IndexedNode, x: List[float], EU: Dict[int, List[str]]) -> int:
    while node["type"] == "internal":
        feat = node["feature_idx"]
        thr = float(EU[feat][node["threshold_pos"]])
        node = node["low"] if x[feat] <= thr else node["high"]
    return node["label_idx"]
