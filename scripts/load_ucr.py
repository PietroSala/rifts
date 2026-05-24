"""UCR .ts loader returning numpy arrays (point-feature view).

Each time-series is treated as a flat vector of point features, consistent with
the scope of the TIME 2026 paper (RF trained on points, no derived features).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from aeon.datasets import load_from_tsfile

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
DATASET_ROOT = DATA_ROOT

# Datasets excluded from the experimental scope. Two families:
#   (a) variable-length or missing-value series — point-feature RF needs
#       fixed length, and the TSF reference does not cover them.
#   (b) datasets on which our HPO sweep failed silently and that we have
#       chosen not to investigate further (out of scope).
EXCLUDED = frozenset({
    # (a) variable-length / missing values
    "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend",
    "Fungi",
    "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2",
    "MelbournePedestrian",
    "PickupGestureWiimoteZ",
    "PLAID",
    "ShakeGestureWiimoteZ",
    # (b) failed during HPO sweep
    "PigAirwayPressure", "PigArtPressure", "PigCVP",
})


def list_datasets() -> list[str]:
    return sorted(
        p.name
        for p in DATASET_ROOT.iterdir()
        if p.is_dir() and p.name not in EXCLUDED
    )


def load_split(name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load TRAIN or TEST split as (X, y) with X shape (n_samples, length)."""
    assert split in ("TRAIN", "TEST")
    path = DATASET_ROOT / name / f"{name}_{split}.ts"
    X, y = load_from_tsfile(str(path))
    # aeon returns (n, 1, length) for univariate; squeeze the channel axis.
    if X.ndim == 3:
        X = X[:, 0, :]
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    return X, y


def load_dataset(name: str) -> dict:
    X_train, y_train = load_split(name, "TRAIN")
    X_test, y_test = load_split(name, "TEST")
    return {
        "name": name,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "length": X_train.shape[1],
        "n_classes": len(np.unique(y_train)),
    }
