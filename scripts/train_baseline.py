"""Tune one Random Forest per UCR dataset with Optuna.

Objective: 5-fold stratified CV accuracy on TRAIN. Final TEST accuracy is
reported once on the held-out TEST split with the best params refit on full
TRAIN. Search space is kept deliberately small so the Endpoint Universe of
the fitted forest stays tractable for the corridor algorithm.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from load_ucr import load_dataset

from _paths import STATE_ROOT as REPO_ROOT, DATA_ROOT  # noqa: E402
HPO_DIR = REPO_ROOT / "hpo"
MODELS_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "metrics"

N_TRIALS = 50
TIMEOUT_SECONDS = 600
CV_SPLITS = 5
SEED = 0

DEPTH_CHOICES = [4, 6, 8, 10, 12]
N_ESTIMATORS_CHOICES = [50, 100, 150, 200]
MIN_SAMPLES_LEAF_CHOICES = [1, 2, 5]
MAX_FEATURES_CHOICES = ["sqrt", "log2", 0.2, 0.3, 0.5, 0.7, 1.0]
CLASS_WEIGHT_CHOICES = [None, "balanced"]


def build_objective(X, y):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", N_ESTIMATORS_CHOICES),
            "max_depth": trial.suggest_categorical("max_depth", DEPTH_CHOICES),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", MIN_SAMPLES_LEAF_CHOICES
            ),
            "max_features": trial.suggest_categorical("max_features", MAX_FEATURES_CHOICES),
            "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHT_CHOICES),
            "criterion": "gini",
            "random_state": SEED,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=1)
        return float(np.mean(scores))

    return objective


def tune_dataset(name: str, n_trials: int = N_TRIALS, timeout: int = TIMEOUT_SECONDS) -> dict:
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(name)
    storage = f"sqlite:///{HPO_DIR / 'optuna.db'}"
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_warmup_steps=5),
    )

    t0 = time.time()
    objective = build_objective(ds["X_train"], ds["y_train"])
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        catch=(Exception,),
    )
    wall_s = time.time() - t0

    best_params = dict(study.best_params)
    final_params = {
        **best_params,
        "criterion": "gini",
        "random_state": SEED,
        "n_jobs": -1,
    }
    clf = RandomForestClassifier(**final_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(ds["X_train"], ds["y_train"])
    test_acc = float(clf.score(ds["X_test"], ds["y_test"]))

    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(
        {"model": clf, "best_params": final_params, "dataset": name},
        model_path,
        compress=3,
    )

    return {
        "dataset": name,
        "n_train": ds["n_train"],
        "n_test": ds["n_test"],
        "length": ds["length"],
        "n_classes": ds["n_classes"],
        "best_params": json.dumps(best_params, sort_keys=True),
        "cv_acc": float(study.best_value),
        "test_acc": test_acc,
        "n_trials_completed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "wall_s": wall_s,
        "model_path": str(model_path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="UCR dataset name (e.g. Adiac)")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    args = parser.parse_args()
    result = tune_dataset(args.dataset, n_trials=args.n_trials, timeout=args.timeout)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
