"""Ticketed caches for the v1 majority-check verifier — F / G / B / C.

Four caches:

  * ``GoodCache`` (G)
      Class-keyed by ``c⋆``. Stores partial α° certifying ``Adv_max(α°, c⋆) ≤ 0``
      (every completion has ``c⋆`` winning). Hit on ``α ⊐ α°`` (incoming α
      specialises cached entry). Antichain in specialisation order — keep the
      *most general*.

  * ``BadCache`` (B)
      Class-keyed by ``c⋆``. Stores the **closure** ᾱ (the more-specific
      α ∪ δ snapshot) at the moment Adv_min(c) > 0 was certified. Hit on
      ``α° ⊐ α`` — i.e. the cached entry SPECIALISES the query (reverse
      direction, same mechanic as C). Antichain — keep the *most specific*.
      Semantically a hit means "the query α has a known-Bad completion
      (the cached α°)", which under the verifier's automaton dynamics
      means the search through α will eventually reach Bad. Hence the
      hit action is still ``raise BadException`` (B differs from C only
      in the action, not the lookup direction).

  * ``FaultyCache`` (F)
      NOT class-keyed (faultiness is a property of the indexed forest).
      Stores partial α† with ``D(α†) ≡ ⊥``. Hit on ``α ⊐ α†``. Antichain;
      keep the *most general*.

  * ``ContinueCache`` (C)
      NOT class-keyed (continue ⇔ "we don't know yet" is class-free). Stores
      partial α° certifying "neither Good nor Bad nor Faulty so far". Hit
      on ``α° ⊐ α`` — i.e. the cached entry SPECIALISES the query (reverse
      direction). Antichain; keep the *most specific* (more specific entries
      cover strictly more queries under reverse specialisation).

Soundness rule (DOT spec, §"caches are pre-loaded before the flow"):

  * The verifier calls ``cache.refresh()`` once at start of Init.
  * During Init / Step the verifier uses ``cache.lookup(α)`` which is
    *local-only* (no Redis read). This freezes the cache state for the
    duration of the call so a peer's mid-flow discovery cannot change the
    answer.
  * ``cache.insert(α)`` broadcasts to Redis (and pulls fresh peer entries
    before pushing, only for dedup purposes).

Ticketing protocol (shared by all four caches):

  Worker-side
    local_entries  list of PartialAssignment (post-antichain merge)
    local_ticket   int — highest Redis ticket reflected locally

  ``refresh()``       — pull entries with score in ``(local_ticket, r_ticket]``,
                        merge under the cache's dominance rule, update
                        ``local_ticket``.
  ``lookup(α)``       — scan ``local_entries`` only. Hit / Miss.
  ``insert(α)``       — pull updates first (to avoid pushing a now-dominated
                        entry), apply the dominance rule, ``INCR`` ticket,
                        ``ZADD`` entry, update ``local_ticket``.

The Redis log is append-only; pruning happens locally per worker.

Redis key layout (per dataset):

    <ds>:G:<c_star>:ticket   counter
    <ds>:G:<c_star>:entries  sorted set
    <ds>:B:<c_star>:ticket
    <ds>:B:<c_star>:entries
    <ds>:F:ticket            no class key
    <ds>:F:entries
    <ds>:C:ticket            no class key
    <ds>:C:entries
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import redis

from drifts.partial_assignment import PartialAssignment


@dataclass
class CacheKeys:
    """Redis key layout for one cache instance.

    F / C are dataset-wide (``c_star`` must be ``None``). G / B are class-keyed
    (``c_star`` is required). Sample index does NOT enter the key.
    """
    dataset: str
    role: str           # "G" | "F" | "B" | "C"
    c_star: Optional[int] = None

    def __post_init__(self) -> None:
        if self.role in ("G", "B") and self.c_star is None:
            raise ValueError(f"role {self.role!r} requires c_star")
        if self.role in ("F", "C") and self.c_star is not None:
            raise ValueError(f"role {self.role!r} must not carry a c_star")

    @property
    def prefix(self) -> str:
        if self.c_star is None:
            return f"{self.dataset}:{self.role}"
        return f"{self.dataset}:{self.role}:{self.c_star}"

    @property
    def ticket(self) -> str:
        return f"{self.prefix}:ticket"

    @property
    def entries(self) -> str:
        return f"{self.prefix}:entries"


@dataclass
class Hit:
    matched: PartialAssignment


@dataclass
class Miss:
    pass


HitOrMiss = Hit | Miss


# ----- base class: ticket pull + Redis write helpers -------------------------


class _TicketedCache:
    """Shared ticketing and Redis I/O for G / F / B / C caches.

    Subclasses provide the ``_match(α, entry)`` predicate and the
    ``_dominance_merge(new_entries)`` rule.
    """

    def __init__(self, r: redis.Redis, keys: CacheKeys):
        self.r = r
        self.keys = keys
        self.local_entries: List[PartialAssignment] = []
        self.local_ticket: int = 0

    # ----- subclass hooks --------------------------------------------------

    def _match(self, alpha: PartialAssignment, entry: PartialAssignment) -> bool:
        raise NotImplementedError

    def _dominance_merge(self,
                         new_entries: Iterable[PartialAssignment]
                         ) -> List[PartialAssignment]:
        """Merge ``new_entries`` into ``local_entries``, applying the cache's
        dominance / dedup rule. Returns the subset of ``new_entries`` that
        actually survived (the rest were dropped as redundant)."""
        raise NotImplementedError

    # ----- ticket pull ------------------------------------------------------

    def _redis_ticket(self) -> int:
        raw = self.r.get(self.keys.ticket)
        return int(raw) if raw is not None else 0

    def _pull_updates(self) -> List[PartialAssignment]:
        r_ticket = self._redis_ticket()
        if r_ticket <= self.local_ticket:
            return []
        raw = self.r.zrangebyscore(self.keys.entries,
                                   self.local_ticket + 1, r_ticket)
        parsed = [PartialAssignment.from_json(s) for s in raw]
        survived = self._dominance_merge(parsed)
        self.local_ticket = r_ticket
        return survived

    # ----- public API -------------------------------------------------------

    def refresh(self) -> int:
        """Pull updates from Redis. The verifier calls this once before Init."""
        return len(self._pull_updates())

    def lookup(self, alpha: PartialAssignment) -> HitOrMiss:
        """Local-only lookup against the snapshot from the last ``refresh()``."""
        for e in self.local_entries:
            if self._match(alpha, e):
                return Hit(matched=e)
        return Miss()

    def insert(self, alpha: PartialAssignment) -> Optional[int]:
        """Insert ``alpha``. Pulls updates first to avoid emitting a
        now-dominated entry; returns the assigned Redis ticket on a real
        insert, or ``None`` if ``alpha`` was redundant.
        """
        self._pull_updates()
        survived = self._dominance_merge([alpha])
        if not survived or alpha not in survived:
            return None
        ticket = self.r.incr(self.keys.ticket)
        self.r.zadd(self.keys.entries, {alpha.to_json(): ticket})
        self.local_ticket = max(self.local_ticket, ticket)
        return ticket

    def __len__(self) -> int:
        return len(self.local_entries)


# ----- forward-specialisation antichain (G, F) -------------------------------


class _ForwardSpecAntichain(_TicketedCache):
    """G / F share the same lookup ``α ⊐ entry`` and the same dominance
    rule (keep the most general — drop any entry that specialises another).
    """

    def _match(self, alpha: PartialAssignment, entry: PartialAssignment) -> bool:
        return alpha.specializes(entry)

    def _dominance_merge(self,
                         new_entries: Iterable[PartialAssignment]
                         ) -> List[PartialAssignment]:
        survived: List[PartialAssignment] = []
        for ne in new_entries:
            if any(ne.specializes(e) and ne != e for e in self.local_entries):
                continue
            if ne in self.local_entries:
                continue
            self.local_entries = [e for e in self.local_entries
                                  if not (e.specializes(ne) and e != ne)]
            self.local_entries.append(ne)
            survived.append(ne)
        return survived


# ----- reverse-specialisation antichain (B, C) -------------------------------


class _ReverseSpecAntichain(_TicketedCache):
    """B / C share the same lookup ``entry ⊐ α`` (cached MORE specific than
    query) and the same dominance rule (keep the most specific). They differ
    only in the verifier's hit action: B-hit raises Bad, C-hit skips work.
    """

    def _match(self, alpha: PartialAssignment, entry: PartialAssignment) -> bool:
        return entry.specializes(alpha)

    def _dominance_merge(self,
                         new_entries: Iterable[PartialAssignment]
                         ) -> List[PartialAssignment]:
        survived: List[PartialAssignment] = []
        for ne in new_entries:
            if any(e.specializes(ne) and e != ne for e in self.local_entries):
                continue
            if ne in self.local_entries:
                continue
            self.local_entries = [e for e in self.local_entries
                                  if not (ne.specializes(e) and ne != e)]
            self.local_entries.append(ne)
            survived.append(ne)
        return survived


class GoodCache(_ForwardSpecAntichain):
    def __init__(self, r: redis.Redis, dataset: str, c_star: int):
        super().__init__(r, CacheKeys(dataset, "G", c_star))


class FaultyCache(_ForwardSpecAntichain):
    def __init__(self, r: redis.Redis, dataset: str):
        super().__init__(r, CacheKeys(dataset, "F"))


class BadCache(_ReverseSpecAntichain):
    """Stores the closure ᾱ; hit when cached ᾱ° ⊐ query α. Hit action in the
    verifier: raise Bad."""
    def __init__(self, r: redis.Redis, dataset: str, c_star: int):
        super().__init__(r, CacheKeys(dataset, "B", c_star))


class ContinueCache(_ReverseSpecAntichain):
    """Stores the closure ᾱ; hit when cached ᾱ° ⊐ query α. Hit action in the
    verifier: skip OBDD / MILP work and continue with the snapshot."""
    def __init__(self, r: redis.Redis, dataset: str):
        super().__init__(r, CacheKeys(dataset, "C"))


# ----- bundle & maintenance --------------------------------------------------


@dataclass
class WorkerCaches:
    """Bundle of (G, F, B, C) for one (dataset, c_star) — what the verifier
    accepts."""
    G: GoodCache
    F: FaultyCache
    B: BadCache
    C: ContinueCache

    def refresh_all(self) -> dict:
        """Pre-load every cache from Redis. Verifier calls this once at the
        start of Init. Returns counts (for telemetry)."""
        return {
            "G": self.G.refresh(),
            "F": self.F.refresh(),
            "B": self.B.refresh(),
            "C": self.C.refresh(),
        }


def open_caches(r: redis.Redis, dataset: str, c_star: int) -> WorkerCaches:
    return WorkerCaches(
        G=GoodCache(r, dataset, c_star),
        F=FaultyCache(r, dataset),
        B=BadCache(r, dataset, c_star),
        C=ContinueCache(r, dataset),
    )


def wipe_dataset_caches(r: redis.Redis, dataset: str) -> int:
    """Nuke every cache key for the dataset — G + B across all c⋆, F, C."""
    n = 0
    patterns = (
        f"{dataset}:G:*",
        f"{dataset}:B:*",
        f"{dataset}:F:ticket", f"{dataset}:F:entries",
        f"{dataset}:C:ticket", f"{dataset}:C:entries",
    )
    for p in patterns:
        keys = list(r.scan_iter(match=p))
        if keys:
            n += r.delete(*keys) or 0
    return n


def wipe_class_caches(r: redis.Redis, dataset: str, c_star: int) -> int:
    """Nuke only the class-keyed G + B for one c⋆; F and C remain intact."""
    keys = list(r.scan_iter(match=f"{dataset}:G:{c_star}:*"))
    keys += list(r.scan_iter(match=f"{dataset}:B:{c_star}:*"))
    return r.delete(*keys) if keys else 0
