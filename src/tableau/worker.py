"""Tableau B&B worker — claim, verify, commit. One worker per process.

Per ``code/docs/tableau_outline.md`` §17 / §18:

  1. Heartbeat → ``tableau:workers``.
  2. ``claim_top_leaf`` → atomic ZPOPMIN + SET NX EX + ZADD inflight.
  3. If saturated: ``Verify``, then ``commit_closure(reason="saturated")``.
  4. Else if ``ρ_max ≤ best.ρ``: ``commit_closure(reason="bb_pruned")``.
  5. Else: ``Verify(inner, c⋆)``:
       - Good → ``Next``, then ``commit_good(parent, child, new_rho?)``.
       - Bad  → random admissible bipartite shrinkage, then
                ``commit_bad(parent, child_A, child_B)``.
  6. Loop. On nil claim, check the termination predicate (§20).

The verifier hook is a callable ``verify(icf, c_star) → VerifyResult.GOOD |
VerifyResult.BAD``. The default adapter wraps our ``drifts.verifier.Verifier``:
Init Good (absorbing 1 or vacuously Good) → GOOD; ``BadException`` → BAD.
``Init → Continue`` is treated as a soft "drive the walk" — handled by the
adapter so the worker can stay unaware.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import redis

from .cicf import CICF, ICFDict, bipartite_shrinkage, canonical_icf_key, next_icf
from .lua_scripts import LuaScripts
from .redis_layout import TableauKeys
from .rho import rho_cells


log = logging.getLogger(__name__)


class VerifyResult(Enum):
    GOOD = "good"
    BAD = "bad"


VerifyFn = Callable[[ICFDict, int], VerifyResult]
RhoFn = Callable[[ICFDict, Sequence[int]], int]


# ---------------------------------------------------------------------------
# Default verifier adapter
# ---------------------------------------------------------------------------


def make_verifier_adapter(irf, *, conservative: bool = True,
                          rng: Optional[random.Random] = None) -> VerifyFn:
    """Build a ``VerifyFn`` over the indexed forest ``irf``.

    The conservative adapter (the default, and the only safe v1 choice)
    calls ``majority_check`` on the initial α of the ICF and returns

      * GOOD iff ``Adv_max → Good`` (every completion in the corridor
        votes ``c⋆`` under Def-3 lex-first);
      * BAD otherwise.

    ``Adv_max → Good`` is the strict reason predicate of §13 of the
    tableau spec — sound (every GOOD certification is a real reason) but
    not complete (some real reasons fail Adv_max on the initial α and
    would need the OBDD / step machinery to be certified). For the v1
    scaffold we accept the incompleteness; the alternative — treating a
    single fair-coin walk through ``Verifier.step`` as proof — is
    unsound because one walk only checks one completion of the corridor.
    """
    from drifts.milp_majority import majority_check
    from drifts.partial_assignment import PartialAssignment
    from drifts.profile import forest_profile

    n_leaves = irf.n_leaves_per_tree()

    def _verify(icf: ICFDict, c_star: int) -> VerifyResult:
        profile = forest_profile(irf, icf)
        alpha = PartialAssignment.initial_from_profile(profile, n_leaves)
        result = majority_check(irf, profile, alpha, c_star)
        if result.verdict == "good":
            return VerifyResult.GOOD
        return VerifyResult.BAD

    return _verify


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@dataclass
class TableauWorker:
    """One tableau B&B worker pulling leaves off Redis.

    Construction parameters:

      ``r``                Redis client.
      ``keys``             ``TableauKeys(dataset, sample_idx)``.
      ``verify``           ``VerifyFn`` — see ``make_verifier_adapter``.
      ``eu_sizes``         per-feature EU lengths (``irf.EU[i] |·|``).
      ``rho``              ρ — defaults to ``rho_cells``.
      ``rng``              per-worker RNG for shrinkage tie-breaking.
      ``worker_id``        unique per worker; defaults to ``pid:hostname``.
      ``ttl``              claim TTL in seconds; doubles as idle threshold.
      ``poll_interval``    sleep when no leaf is claimable.
      ``gc_every``         run ``gc_inflight`` after this many empty pops.
    """
    r: redis.Redis
    keys: TableauKeys
    verify: VerifyFn
    eu_sizes: Sequence[int]
    rho: RhoFn = field(default=rho_cells)
    rng: random.Random = field(default_factory=random.Random)
    worker_id: str = ""
    ttl: int = 60
    poll_interval: float = 0.5
    gc_every: int = 8

    _scripts: Optional[LuaScripts] = field(default=None, init=False)
    _empty_polls: int = field(default=0, init=False)

    def __post_init__(self):
        if not self.worker_id:
            import os, socket
            self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self._scripts = LuaScripts.bind(self.r)

    # ------------------------------------------------------------------ root

    def seed_root(self, root: CICF) -> None:
        """Initialise the tableau with the given root node. Idempotent."""
        rho_cur = self.rho(root.inner, self.eu_sizes)
        rho_max = self.rho(root.ceiling, self.eu_sizes)
        rho_min = self.rho(root.floor, self.eu_sizes)
        payload = root.to_payload()
        node_id = payload["id"]
        node_key = self.keys.node(node_id)
        pipe = self.r.pipeline()
        pipe.hset(node_key, mapping={
            **payload,
            "rho_cur": rho_cur, "rho_max": rho_max, "rho_min": rho_min,
            "status": "leaf",
        })
        pipe.zadd(self.keys.leaves, {node_id: -rho_max})
        pipe.hsetnx(self.keys.best, "rho", str(rho_cur))
        pipe.hsetnx(self.keys.best, "id", node_id)
        pipe.hsetnx(self.keys.best, "icf", canonical_icf_key(root.inner))
        pipe.set(self.keys.state, "running", nx=True)
        pipe.execute()
        log.info("seeded root %s (rho_cur=%d rho_max=%d rho_min=%d)",
                 node_id, rho_cur, rho_max, rho_min)

    # ------------------------------------------------------------------ main

    def run_forever(self) -> dict:
        """Drive the worker loop until ``tableau:state`` flips to non-running
        (or termination predicate is satisfied). Returns a stats dict."""
        stats = {"verified": 0, "good": 0, "bad": 0, "saturated": 0,
                 "bb_pruned": 0, "gc_reaped": 0, "loops": 0}
        while True:
            stats["loops"] += 1
            self._heartbeat()
            state = self.r.get(self.keys.state)
            if state in (b"done", "done", b"aborted", "aborted"):
                break
            claim = self._claim_top()
            if claim is None:
                self._empty_polls += 1
                if self._empty_polls % self.gc_every == 0:
                    reaped = self._gc_inflight()
                    stats["gc_reaped"] += reaped
                if self._termination_predicate():
                    self.r.set(self.keys.state, "done")
                    break
                time.sleep(self.poll_interval)
                continue
            self._empty_polls = 0
            node_id, _score = claim
            self._process(node_id, stats)
        return stats

    # ------------------------------------------------------------------ low-level

    def _heartbeat(self) -> None:
        self.r.hset(self.keys.workers, self.worker_id, str(int(time.time())))

    def _claim_top(self):
        return self._scripts.claim_top_leaf(
            keys=[self.keys.leaves, self.keys.inflight, self.keys.claim_prefix],
            args=[self.worker_id, self.ttl, int(time.time())],
        )

    def _gc_inflight(self) -> int:
        return int(self._scripts.gc_inflight(
            keys=[self.keys.leaves, self.keys.inflight,
                  f"{self.keys.prefix}:node:", self.keys.claim_prefix],
            args=[int(time.time()), self.ttl],
        ) or 0)

    def _read_best_rho(self) -> Optional[int]:
        raw = self.r.hget(self.keys.best, "rho")
        return int(raw) if raw is not None else None

    def _termination_predicate(self) -> bool:
        n_leaves = self.r.zcard(self.keys.leaves)
        n_inflight = self.r.zcard(self.keys.inflight)
        if n_leaves or n_inflight:
            return False
        # every worker idle for ≥ 2·ttl?
        now = int(time.time())
        workers = self.r.hgetall(self.keys.workers) or {}
        cutoff = now - 2 * self.ttl
        for _w, ts in workers.items():
            try:
                if int(ts) > cutoff:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    # ------------------------------------------------------------------ leaf processing

    def _process(self, node_id: str, stats: dict) -> None:
        node_key = self.keys.node(node_id)
        raw = self.r.hgetall(node_key)
        if not raw:
            log.warning("processing missing node %s", node_id)
            return
        # decode_responses on the client may return bytes/strs depending on config
        payload = {k.decode() if isinstance(k, bytes) else k:
                   v.decode() if isinstance(v, bytes) else v
                   for k, v in raw.items()}
        try:
            c = CICF.from_payload(payload)
        except Exception as e:
            log.warning("malformed node %s: %s", node_id, e)
            return

        # 1. Saturation
        if c.is_expansion_saturated():
            stats["verified"] += 1
            verdict = self.verify(c.inner, c.class_label)
            stats["saturated"] += 1
            new_rho = self.rho(c.inner, self.eu_sizes) if verdict == VerifyResult.GOOD else None
            self._commit_closure(node_id, "saturated",
                                 new_rho=new_rho,
                                 inner_canon=canonical_icf_key(c.inner))
            if verdict == VerifyResult.GOOD:
                stats["good"] += 1
            else:
                stats["bad"] += 1
            return

        # 2. B&B prune
        rho_max = int(payload["rho_max"])
        best_rho = self._read_best_rho()
        if best_rho is not None and rho_max <= best_rho:
            stats["bb_pruned"] += 1
            self._commit_closure(node_id, "bb_pruned")
            return

        # 3. Verify
        stats["verified"] += 1
        verdict = self.verify(c.inner, c.class_label)
        if verdict == VerifyResult.GOOD:
            stats["good"] += 1
            child = next_icf(c, self.eu_sizes)
            new_rho = self.rho(c.inner, self.eu_sizes)
            self._commit_good(node_id, child, parent_new_rho=new_rho,
                              parent_inner_canon=canonical_icf_key(c.inner))
        else:
            stats["bad"] += 1
            res = bipartite_shrinkage(c, self.eu_sizes, self.rng)
            if res is None:
                # No shrinkage admissible — treat as shrink-saturated leaf,
                # close as bad.
                self._commit_closure(node_id, "bb_pruned")
                return
            child_a, child_b, _f, _d = res
            self._commit_bad(node_id, child_a, child_b)

    # ------------------------------------------------------------------ commit wrappers

    def _commit_closure(self, node_id: str, reason: str,
                        new_rho: Optional[int] = None,
                        inner_canon: str = "") -> None:
        self._scripts.commit_closure(
            keys=[
                self.keys.inflight, self.keys.internals, self.keys.best,
                f"{self.keys.prefix}:node:", self.keys.claim_prefix,
            ],
            args=[
                node_id, reason,
                "" if new_rho is None else str(new_rho),
                inner_canon,
            ],
        )

    def _commit_good(self, parent_id: str, child: CICF, *,
                     parent_new_rho: int, parent_inner_canon: str) -> None:
        child_payload = child.to_payload()
        child_id = child_payload["id"]
        rho_cur = self.rho(child.inner, self.eu_sizes)
        rho_max = self.rho(child.ceiling, self.eu_sizes)
        rho_min = self.rho(child.floor, self.eu_sizes)
        self._scripts.commit_good(
            keys=[
                self.keys.leaves, self.keys.inflight, self.keys.internals,
                self.keys.best, f"{self.keys.prefix}:node:",
                self.keys.claim_prefix,
            ],
            args=[
                parent_id, child_id, json.dumps(child_payload),
                str(rho_max), str(rho_cur), str(rho_min),
                str(parent_new_rho), parent_inner_canon,
            ],
        )

    def _commit_bad(self, parent_id: str, child_a: CICF, child_b: CICF) -> None:
        a = child_a.to_payload(); b = child_b.to_payload()
        a_id = a["id"]; b_id = b["id"]
        a_rho = (self.rho(child_a.inner, self.eu_sizes),
                 self.rho(child_a.ceiling, self.eu_sizes),
                 self.rho(child_a.floor, self.eu_sizes))
        b_rho = (self.rho(child_b.inner, self.eu_sizes),
                 self.rho(child_b.ceiling, self.eu_sizes),
                 self.rho(child_b.floor, self.eu_sizes))
        self._scripts.commit_bad(
            keys=[
                self.keys.leaves, self.keys.inflight, self.keys.internals,
                f"{self.keys.prefix}:node:", self.keys.claim_prefix,
            ],
            args=[
                parent_id,
                a_id, json.dumps(a), str(a_rho[1]), str(a_rho[0]), str(a_rho[2]),
                b_id, json.dumps(b), str(b_rho[1]), str(b_rho[0]), str(b_rho[2]),
            ],
        )
