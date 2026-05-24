"""Def-3-compliant prediction wrapper around a fitted sklearn forest.

`Def3Forest` exposes the same `predict(X)` interface as
`sklearn.ensemble.RandomForestClassifier` but uses **pure majority voting
over per-tree argmax** with the **lex-first tie-break** specified in
`docs/follow.md`. The wrapper does not retrain anything; it merely walks
the trees that sklearn already fitted.

Why a wrapper rather than sklearn's `predict()`. sklearn averages
`predict_proba` across trees and takes argmax, which silently differs from
the paper's Definition 3 whenever leaves are non-pure (e.g. `min_samples_leaf > 1`).
Our IndexedRandomForest implements Def 3 strictly, so the test uses
`Def3Forest` as the reference oracle.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class Def3Forest:
    """Reference Def-3 forest semantics on top of a fitted sklearn RF.

    Tie-break conventions (see docs/follow.md):
      - **Leaf labelling.** Argmax over `tree.value[node]`; ties broken by
        lowest sklearn class index, which (since `sklearn.classes_` is sorted)
        equals the lex-first class label.
      - **Forest vote.** Argmax over per-class vote counts; ties broken by
        lex-first class label.
    """

    def __init__(self, rf):
        self.rf = rf
        # sklearn keeps these sorted; use canonical string keys to avoid
        # surprises from estimators whose per-tree `classes_` is a subset of
        # the forest's (e.g. a class absent from a bootstrap).
        self.classes_ = list(rf.classes_)
        self.class_keys = [str(c) for c in self.classes_]
        # canonical label per tree: leaf argmax of tree.value, using the
        # forest's class index space (Def 3 + lex-first tie-break per docs/follow.md).
        self._leaf_labels_per_tree = [
            np.array([str(rf.classes_[int(np.argmax(v[0]))])
                      for v in est.tree_.value])
            for est in rf.estimators_
        ]

    def per_tree_predict(self, X: np.ndarray) -> np.ndarray:
        """Shape `(n_trees, n_samples)`. Each cell is the leaf class label
        (as a string) under leaf-argmax with first-index (= lex-first) tie-break."""
        X = np.asarray(X, dtype=np.float32)
        out = []
        for est, leaf_labels in zip(self.rf.estimators_, self._leaf_labels_per_tree):
            leaf_ids = est.apply(X)
            out.append(leaf_labels[leaf_ids])
        return np.array(out)

    def predict(self, X: np.ndarray) -> np.ndarray:
        per_tree = self.per_tree_predict(np.asarray(X))
        out = []
        for col in per_tree.T:
            counts = {c: 0 for c in self.class_keys}
            for v in col:
                counts[str(v)] += 1
            # max with key returns the first occurrence of the maximum.
            # `self.class_keys` is in sklearn's class order (sorted), so this
            # is the lex-first winner on ties.
            out.append(max(self.class_keys, key=lambda c: counts[c]))
        return np.array(out)
