"""Save/load the IndexedRandomForest and trivial ICFs in DB 0 (`DATA`).

Key layout (all under a per-dataset prefix `<ds>:…`):

    <ds>:IRF:meta                 string   JSON {n_trees, n_features, n_labels}
    <ds>:IRF:phi_F                hash     feature_name -> feature_idx
    <ds>:IRF:phi_L                hash     label_name -> label_idx
    <ds>:IRF:EU:<i>               string   JSON [repr(float), …] (sorted asc)
    <ds>:IRF:tree:<j>             string   JSON of indexed tree (recursive)
    <ds>:sample:<k>:ICF:trivial   string   JSON {feature_idx: [b_pos, e_pos]}

Floats are persisted as strings (`repr(float)`) for bit-exact round-trip.
"""
from __future__ import annotations

import json
from typing import List

import redis

from drifts.icf import ICFIndexed
from drifts.indexed_forest import IndexedRandomForest


def _k(ds: str, *parts) -> str:
    return ":".join((ds, *map(str, parts)))


# ---------- indexed forest ----------------------------------------------------


def save_indexed_forest(r: redis.Redis, irf: IndexedRandomForest) -> None:
    ds = irf.dataset
    if not ds:
        raise ValueError("IndexedRandomForest.dataset is empty; refusing to write")
    pipe = r.pipeline(transaction=False)
    pipe.set(_k(ds, "IRF", "meta"), json.dumps({
        "dataset": ds,
        "n_trees": irf.n_trees,
        "n_features": irf.n_features,
        "n_labels": irf.n_labels,
    }))
    pipe.delete(_k(ds, "IRF", "phi_F"))
    pipe.hset(_k(ds, "IRF", "phi_F"), mapping={k: str(v) for k, v in irf.phi_F.items()})
    pipe.delete(_k(ds, "IRF", "phi_L"))
    pipe.hset(_k(ds, "IRF", "phi_L"), mapping={k: str(v) for k, v in irf.phi_L.items()})
    for i in range(irf.n_features):
        pipe.set(_k(ds, "IRF", "EU", i), json.dumps(irf.EU[i]))
    for j, tree in enumerate(irf.trees):
        pipe.set(_k(ds, "IRF", "tree", j), json.dumps(tree))
    pipe.execute()


def load_indexed_forest(r: redis.Redis, dataset: str) -> IndexedRandomForest:
    meta_raw = r.get(_k(dataset, "IRF", "meta"))
    if meta_raw is None:
        raise KeyError(f"no IndexedRandomForest stored for dataset {dataset!r}")
    meta = json.loads(meta_raw)
    phi_F = {k: int(v) for k, v in r.hgetall(_k(dataset, "IRF", "phi_F")).items()}
    phi_L = {k: int(v) for k, v in r.hgetall(_k(dataset, "IRF", "phi_L")).items()}
    EU = {}
    for i in range(meta["n_features"]):
        raw = r.get(_k(dataset, "IRF", "EU", i))
        EU[i] = json.loads(raw) if raw is not None else []
    trees: List = []
    for j in range(meta["n_trees"]):
        raw = r.get(_k(dataset, "IRF", "tree", j))
        if raw is None:
            raise KeyError(f"missing tree {j} for {dataset!r}")
        trees.append(json.loads(raw))
    return IndexedRandomForest(
        dataset=dataset, phi_F=phi_F, phi_L=phi_L, EU=EU, trees=trees,
    )


def delete_indexed_forest(r: redis.Redis, dataset: str) -> int:
    """Wipe the indexed forest for `dataset`. Returns the number of keys removed."""
    keys = list(r.scan_iter(match=_k(dataset, "IRF", "*")))
    return r.delete(*keys) if keys else 0


# ---------- trivial ICF -------------------------------------------------------


def save_trivial_icf(r: redis.Redis, dataset: str, sample_idx: int,
                     icf: ICFIndexed) -> None:
    payload = {str(i): list(pair) for i, pair in icf.items()}
    r.set(_k(dataset, "sample", sample_idx, "ICF", "trivial"), json.dumps(payload))


def load_trivial_icf(r: redis.Redis, dataset: str, sample_idx: int) -> ICFIndexed:
    raw = r.get(_k(dataset, "sample", sample_idx, "ICF", "trivial"))
    if raw is None:
        raise KeyError(f"no trivial ICF stored for {dataset!r} sample {sample_idx}")
    payload = json.loads(raw)
    return {int(k): tuple(v) for k, v in payload.items()}
