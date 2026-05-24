"""Ticketed F / G / B / C caches: ticketing across workers, class-keyed
isolation, cross-sample sharing within a (dataset, c⋆).

Soundness rule: ``lookup()`` is local-only (no Redis read mid-flow); the
verifier calls ``refresh()`` once before Init to pre-load the snapshot.
``insert()`` still pulls before pushing (dedup).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1] / "src"
sys.path.insert(0, str(SRC))

from cache.caches import (
    BadCache, ContinueCache, FaultyCache, GoodCache, Hit, Miss,
    open_caches, wipe_class_caches, wipe_dataset_caches,
)
from cache.connection import get_client
from drifts.partial_assignment import PartialAssignment

DATASET = "_test_ticketed_caches"
C_STAR_A = 0
C_STAR_B = 1


def _pa(decided, n_leaves_per_tree=()):
    p = PartialAssignment(n_leaves_per_tree=tuple(n_leaves_per_tree))
    for t, m in decided.items():
        for v, val in m.items():
            p.set(t, v, val)
    return p


def test_good_cache_ticketing_and_class_isolation():
    r = get_client("DATA")
    wipe_dataset_caches(r, DATASET)

    A = GoodCache(r, DATASET, C_STAR_A)
    B = GoodCache(r, DATASET, C_STAR_A)
    OTHER = GoodCache(r, DATASET, C_STAR_B)

    # G is forward-spec: lookup hits when α ⊐ entry.
    alpha_gen = _pa({0: {5: 0}})
    tk = A.insert(alpha_gen)
    assert tk == 1

    B.refresh()
    alpha_spec = _pa({0: {5: 0, 9: 1}})
    res = B.lookup(alpha_spec)
    assert isinstance(res, Hit), f"expected Hit, got {res!r}"

    OTHER.refresh()
    res = OTHER.lookup(alpha_spec)
    assert isinstance(res, Miss)

    OTHER.insert(_pa({1: {0: 1}}))
    A.refresh()
    res = A.lookup(_pa({1: {0: 1, 2: 0}}))
    assert isinstance(res, Miss)

    wipe_dataset_caches(r, DATASET)
    print("good-cache ticketing + class isolation OK")


def test_cross_sample_sharing_in_dataset():
    """G / B / F / C entries persist across samples within the same
    (dataset, c_star). Fresh handles must see prior inserts after refresh()."""
    r = get_client("DATA")
    wipe_dataset_caches(r, DATASET)

    s0 = open_caches(r, DATASET, C_STAR_A)
    s0.G.insert(_pa({0: {5: 0}}))
    s0.B.insert(_pa({0: {2: 1}, 1: {0: 1}}))
    s0.F.insert(_pa({2: {0: 0, 1: 0, 2: 0}}))
    s0.C.insert(_pa({0: {5: 0, 9: 1}}))

    s17 = open_caches(r, DATASET, C_STAR_A)
    s17.refresh_all()
    assert isinstance(s17.G.lookup(_pa({0: {5: 0, 7: 1}})), Hit)
    assert isinstance(s17.B.lookup(_pa({0: {2: 1, 3: 0}, 1: {0: 1}})), Hit)
    assert isinstance(s17.F.lookup(_pa({2: {0: 0, 1: 0, 2: 0, 3: 0}})), Hit)
    # C: cached α° = {0:{5:0, 9:1}} hits query α = {0:{5:0}} because α° ⊐ α
    assert isinstance(s17.C.lookup(_pa({0: {5: 0}})), Hit)

    wipe_dataset_caches(r, DATASET)
    print("cross-sample sharing OK")


def test_faulty_cache_dataset_wide():
    """F is shared across all c_star within the same dataset."""
    r = get_client("DATA")
    wipe_dataset_caches(r, DATASET)

    F0 = FaultyCache(r, DATASET)
    F1 = FaultyCache(r, DATASET)
    F0.insert(_pa({0: {0: 0, 1: 0}}))
    F1.refresh()
    assert isinstance(F1.lookup(_pa({0: {0: 0, 1: 0, 2: 0}})), Hit)

    wipe_dataset_caches(r, DATASET)
    print("faulty-cache dataset-wide OK")


def test_continue_cache_reverse_dominance():
    """C: cached α° hits the query α when α° ⊐ α (entry specialises the query).
    Keep the *most specific* — newer-and-more-specific drops older-and-more-general.
    """
    r = get_client("DATA")
    wipe_dataset_caches(r, DATASET)

    C0 = ContinueCache(r, DATASET)
    C0.insert(_pa({0: {5: 0}}))                 # less specific
    C0.insert(_pa({0: {5: 0, 9: 1}}))           # strictly more specific
    # Local snapshot keeps only the more-specific entry.
    assert len(C0) == 1, f"expected 1 entry, got {len(C0)}"

    # Query: a STRICTLY more general α (less specific than any cached). Hit.
    assert isinstance(C0.lookup(_pa({0: {5: 0}})), Hit)
    # Query: constrains a tree the cache says nothing about. Miss.
    assert isinstance(C0.lookup(_pa({1: {0: 0}})), Miss)

    # Fresh handle picks up via refresh
    C1 = ContinueCache(r, DATASET)
    C1.refresh()
    assert isinstance(C1.lookup(_pa({0: {5: 0}})), Hit)

    wipe_dataset_caches(r, DATASET)
    print("continue-cache reverse dominance OK")


def test_wipe_class_caches_isolates_one_c_star():
    """wipe_class_caches(c*) clears G/B for c* only; F + C stay intact, other
    c_star's G/B stay intact."""
    r = get_client("DATA")
    wipe_dataset_caches(r, DATASET)

    GA = GoodCache(r, DATASET, C_STAR_A); GA.insert(_pa({0: {0: 0}}))
    GB = GoodCache(r, DATASET, C_STAR_B); GB.insert(_pa({0: {1: 0}}))
    F  = FaultyCache(r, DATASET);          F.insert(_pa({1: {0: 0, 1: 0}}))
    C  = ContinueCache(r, DATASET);        C.insert(_pa({2: {0: 0, 3: 1}}))

    wipe_class_caches(r, DATASET, C_STAR_A)

    # Fresh handles + refresh — observe what survived.
    g_a = GoodCache(r, DATASET, C_STAR_A); g_a.refresh()
    g_b = GoodCache(r, DATASET, C_STAR_B); g_b.refresh()
    f_  = FaultyCache(r, DATASET);          f_.refresh()
    c_  = ContinueCache(r, DATASET);        c_.refresh()

    assert isinstance(g_a.lookup(_pa({0: {0: 0}})), Miss)
    assert isinstance(g_b.lookup(_pa({0: {1: 0}})), Hit)
    assert isinstance(f_.lookup(_pa({1: {0: 0, 1: 0}})), Hit)
    assert isinstance(c_.lookup(_pa({2: {0: 0}})), Hit)

    wipe_dataset_caches(r, DATASET)
    print("wipe-class-caches isolation OK")


def test_bad_cache_reverse_specialization_lookup():
    """B stores the closure ᾱ; hit when cached α° ⊐ query α (reverse spec,
    same mechanic as C, action on hit differs — verifier raises Bad).
    Antichain: keep the most specific."""
    r = get_client("DATA")
    wipe_dataset_caches(r, DATASET)

    A = BadCache(r, DATASET, C_STAR_A)
    B = BadCache(r, DATASET, C_STAR_A)

    A.insert(_pa({0: {5: 0}}))                   # less specific
    A.insert(_pa({0: {5: 0, 9: 1}}))             # strictly more specific — dominates
    assert len(A) == 1, f"expected 1 entry after antichain, got {len(A)}"

    B.refresh()
    # Query is STRICTLY more general than the cached more-specific entry → Hit.
    assert isinstance(B.lookup(_pa({0: {5: 0}})), Hit)
    # Query constrains a tree the cache says nothing about → Miss.
    assert isinstance(B.lookup(_pa({1: {0: 0}})), Miss)

    wipe_dataset_caches(r, DATASET)
    print("bad-cache reverse spec OK")


if __name__ == "__main__":
    test_good_cache_ticketing_and_class_isolation()
    test_cross_sample_sharing_in_dataset()
    test_faulty_cache_dataset_wide()
    test_continue_cache_reverse_dominance()
    test_wipe_class_caches_isolates_one_c_star()
    test_bad_cache_reverse_specialization_lookup()
    print("OK")
