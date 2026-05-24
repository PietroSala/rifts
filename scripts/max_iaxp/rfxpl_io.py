"""Materialise the inputs RFxpl/GiAXp expects.

RFxpl's RFSklearn loads a sklearn RandomForestClassifier via plain pickle.load.
Its Dataset class reads a CSV with `f0,f1,...,f{L-1},class` header where the
last column is the class label.

We write only the train portion to the data CSV; bounds (x_min, x_max) for
INFXRF.explain are computed from that same array.
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


def write_model_pkl(rf, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(rf, f)
    return path


def write_data_csv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    path: Path,
) -> Tuple[Path, np.ndarray, np.ndarray]:
    """Write the TRAIN portion as RFxpl-compatible CSV.

    Returns (path, x_min_per_feature, x_max_per_feature) — the latter two are
    used as x_bounds for INFXRF.explain.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_features = X_train.shape[1]
    header = [f"f{i}" for i in range(n_features)] + ["class"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row, label in zip(X_train, y_train):
            w.writerow([f"{v:.6g}" for v in row] + [str(label)])
    x_min = np.min(X_train, axis=0).astype(np.float32)
    x_max = np.max(X_train, axis=0).astype(np.float32)
    return path, x_min, x_max
