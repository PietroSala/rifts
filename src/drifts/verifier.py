"""Verifier — Init / Reset / Step per ``code/design/dot/verifier.dot``.

The verifier is a teacher for an L*-style automata-learning loop.
``Init(icf)`` builds the initial state from an ICF; ``Reset()`` restores the
Init snapshot; ``Step(σ)`` consumes one symbol (``ε / 0 / 1``).

Soundness rule: caches are pre-loaded once via ``caches.refresh_all()`` at
the start of ``Init``; ``lookup()`` is local-only thereafter; emissions
broadcast to Redis.
"""
from __future__ import annotations

import random
from copy import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .indexed_forest import IndexedRandomForest
from .milp_majority import majority_check, majority_check_min
from .obdd import OBDDContext
from .partial_assignment import PartialAssignment
from .profile import forest_profile


EPSILON = "ε"
EmittedLabel = int  # 0 | 1
RhoEntry = Tuple[int, int]  # (tree_idx, leaf_idx)


class BadException(Exception):
    """Raised when the verifier discovers a Bad state (an extension exists
    that beats ``c⋆`` under Def-3 lex-first). Stops the learning loop.
    """
    def __init__(self, alpha: PartialAssignment, *, reason: str = ""):
        super().__init__(reason or "Bad state discovered")
        self.alpha = alpha
        self.reason = reason


@dataclass
class _Snapshot:
    alpha: PartialAssignment
    rho: List[RhoEntry]
    D: Optional[Any]            # _cudd.Function | None
    absorbing: Optional[int]    # None | 0 | 1
    profile: List[set]          # cached: needed by Step's MILP calls


@dataclass
class Verifier:
    """Stateful verifier for one (RF, OBDD, caches, c⋆) triple.

    Lifecycle:

      v = Verifier(irf, ctx, caches, c_star)
      v.init(icf)         # → may set absorbing or raise BadException
      v.step(σ)           # → 0 / 1 / raises BadException
      v.reset()           # restore the Init snapshot
    """
    irf: IndexedRandomForest
    ctx: OBDDContext
    caches: Any                          # WorkerCaches
    c_star: int
    rng: random.Random = field(default_factory=random.Random)

    # ----- mutable state (populated by init / step) ------------------------
    alpha: Optional[PartialAssignment] = field(default=None, init=False)
    init_alpha: Optional[PartialAssignment] = field(default=None, init=False)
    rho: List[RhoEntry] = field(default_factory=list, init=False)
    D: Optional[Any] = field(default=None, init=False)
    absorbing: Optional[int] = field(default=None, init=False)
    profile: List[set] = field(default_factory=list, init=False)
    snapshot: Optional[_Snapshot] = field(default=None, init=False)

    # ======================================================================
    # Init
    # ======================================================================

    def init(self, icf) -> Optional[int]:
        """Run the Init cascade per the DOT. Returns the absorbing label
        (``0`` / ``1``) if Init determined it, ``None`` if Init landed in
        the undetermined Continue state. Raises ``BadException`` if Init
        discovered Bad.
        """
        self.caches.refresh_all()

        self.profile = list(forest_profile(self.irf, icf))
        n_leaves = self.irf.n_leaves_per_tree()
        alpha = PartialAssignment.initial_from_profile(self.profile, n_leaves)
        self.alpha = alpha
        self.init_alpha = alpha           # snapshot for AAL-terminal emissions
        self.D = None
        self.absorbing = None
        self.rho = []

        # F → G → B → C — pre-loaded local lookups
        if self._hit(self.caches.F, alpha):
            self.absorbing = 0
            self._snap(); return 0
        if self._hit(self.caches.G, alpha):
            self.absorbing = 1
            self._snap(); return 1
        if self._hit(self.caches.B, alpha):
            self._snap(); raise BadException(alpha, reason="B-cache hit at Init")
        if self._hit(self.caches.C, alpha):
            self.rho = self._build_rho(alpha)
            self._snap(); return None

        # Trivial-slackless: every tree determined (canonical: one 1-mark per tree)
        if alpha.is_complete(n_leaves):
            winner = self._vote(alpha)
            if winner == self.c_star:
                self.caches.G.insert(alpha)
                self.absorbing = 1
                self._snap(); return 1
            self.caches.B.insert(alpha)
            self._snap()
            raise BadException(alpha, reason="trivial-slackless: c⋆ loses vote at Init")

        # Build D
        self.D = self.ctx.build_D(alpha)
        if self.ctx.is_corridor_unsat(self.D):
            self.caches.F.insert(alpha)
            self.absorbing = 0
            self._snap(); return 0

        # δ, ᾱ
        delta = self.ctx.compute_delta(alpha, self.D)
        oalpha = self._merge(alpha, delta)

        # Adv_max → Good
        res_max = majority_check(self.irf, self.profile, oalpha, self.c_star)
        if res_max.verdict == "good":
            self.caches.G.insert(alpha)        # NB: α (smaller), not ᾱ
            self.absorbing = 1
            self._snap(); return 1

        # Adv_min → Bad
        res_min = majority_check_min(self.irf, self.profile, oalpha, self.c_star)
        if res_min.verdict == "bad":
            self.caches.B.insert(oalpha)       # NB: closure ᾱ (B is reverse-spec)
            self._snap()
            raise BadException(alpha, reason="Adv_min Bad at Init")

        # Continue: emit ᾱ (bigger) to C, build ρ from ᾱ, snapshot
        self.caches.C.insert(oalpha)
        self.alpha = oalpha
        self.rho = self._build_rho(oalpha)
        self._snap()
        return None

    # ======================================================================
    # Reset
    # ======================================================================

    def reset(self) -> None:
        """Restore the snapshot captured at the end of Init."""
        if self.snapshot is None:
            raise RuntimeError("reset() before init()")
        s = self.snapshot
        self.alpha = copy(s.alpha)
        self.rho = list(s.rho)
        self.D = s.D
        self.absorbing = s.absorbing
        self.profile = s.profile

    # ======================================================================
    # Step
    # ======================================================================

    def step(self, sigma) -> EmittedLabel:
        """Consume one symbol ``σ ∈ {ε, 0, 1}``.

          ``ε``  →  returns the absorbing label if set, else ``0``.
          ``0/1``→  commits ρ[0] := σ, runs the cache/D/MILP cascade, may set
                    absorbing or raise BadException. Always returns ``0`` on
                    successful (non-Bad) transition; the caller queries ε for
                    the absorbing label.
        """
        if sigma == EPSILON or sigma is None:
            return self.absorbing if self.absorbing is not None else 0

        if sigma not in (0, 1):
            raise ValueError(f"sigma must be ε / 0 / 1, got {sigma!r}")

        if self.absorbing is not None:
            return self.absorbing

        if not self.rho:
            raise RuntimeError(
                "rho is empty but absorbing not set — protocol violation"
            )

        h_tree, h_leaf = self.rho[0]
        alpha_new = self._with_set(self.alpha, h_tree, h_leaf, sigma)

        # Cache fast paths (no D needed)
        if self._hit(self.caches.F, alpha_new):
            self.caches.F.insert(alpha_new)
            self.alpha = alpha_new
            self.absorbing = 0
            return 0
        if self._hit(self.caches.G, alpha_new):
            self.alpha = alpha_new
            self.absorbing = 1
            return 0
        if self._hit(self.caches.C, alpha_new):
            self.alpha = alpha_new
            self._prune_rho()
            return 0

        # Extend / build D, then check D ≡ ⊥
        if self.D is None:
            self.D = self.ctx.build_D(alpha_new)
        else:
            self.D = self.ctx.conjoin_D(self.D, h_tree, h_leaf, sigma)

        if self.ctx.is_corridor_unsat(self.D):
            self.caches.F.insert(alpha_new)
            self.alpha = alpha_new
            self.absorbing = 0
            return 0

        # B-check is safe NOW (D ≢ ⊥ confirmed)
        if self._hit(self.caches.B, alpha_new):
            raise BadException(alpha_new, reason="B-cache hit at Step (D ≢ ⊥)")

        # δ, ᾱ
        delta = self.ctx.compute_delta(alpha_new, self.D)
        oalpha = self._merge(alpha_new, delta)

        # Trivial-slackless
        n_leaves = self.irf.n_leaves_per_tree()
        if oalpha.is_complete(n_leaves):
            winner = self._vote(oalpha)
            if winner == self.c_star:
                # AAL-terminal rule: push the INIT profile to G (the call's α).
                self.caches.G.insert(self.init_alpha)
                self.alpha = oalpha
                self.absorbing = 1
                return 0
            # closure for the slackless Bad is the slackless α itself
            self.caches.B.insert(oalpha)
            raise BadException(alpha_new, reason="trivial-slackless: c⋆ loses at Step")

        # Adv_max → Good
        res_max = majority_check(self.irf, self.profile, oalpha, self.c_star)
        if res_max.verdict == "good":
            # AAL-terminal rule: push the INIT profile to G.
            self.caches.G.insert(self.init_alpha)
            self.alpha = oalpha
            self.absorbing = 1
            return 0

        # Adv_min → Bad
        res_min = majority_check_min(self.irf, self.profile, oalpha, self.c_star)
        if res_min.verdict == "bad":
            self.caches.B.insert(oalpha)       # NB: closure ᾱ (B is reverse-spec)
            raise BadException(alpha_new, reason="Adv_min Bad at Step")

        # Continue: emit ᾱ to C, advance state, prune ρ
        self.caches.C.insert(oalpha)
        self.alpha = oalpha
        self._prune_rho()
        return 0

    # ======================================================================
    # internal helpers
    # ======================================================================

    def _hit(self, cache, alpha) -> bool:
        from cache.caches import Hit
        return isinstance(cache.lookup(alpha), Hit)

    def _snap(self) -> None:
        self.snapshot = _Snapshot(
            alpha=copy(self.alpha),
            rho=list(self.rho),
            D=self.D,
            absorbing=self.absorbing,
            profile=list(self.profile),
        )

    def _with_set(self, alpha: PartialAssignment,
                  t: int, v: int, val: int) -> PartialAssignment:
        """Return a fresh canonical α with (t, v) := val committed."""
        new_decided = {tt: dict(mm) for tt, mm in alpha.decided.items()}
        new = PartialAssignment(decided=new_decided,
                                n_leaves_per_tree=alpha.n_leaves_per_tree)
        new.set(t, v, val)
        return new

    def _merge(self, alpha: PartialAssignment,
               delta: PartialAssignment) -> PartialAssignment:
        """ᾱ := canonical(α ∪ δ). Construction triggers ``__post_init__``
        canonicalisation, which also applies the all-but-one promotion when
        δ adds the (n_t − 1)-th 0-mark in a tree.
        """
        merged: dict = {tt: dict(mm) for tt, mm in alpha.decided.items()}
        for t, m in delta.decided.items():
            slot = merged.setdefault(t, {})
            for v, val in m.items():
                existing = slot.get(v)
                if existing is not None and existing != val:
                    raise ValueError(
                        f"merge conflict on (tree={t}, leaf={v}): "
                        f"α={existing}, δ={val}"
                    )
                slot[v] = val
        return PartialAssignment(decided=merged,
                                 n_leaves_per_tree=self.irf.n_leaves_per_tree())

    def _build_rho(self, alpha: PartialAssignment) -> List[RhoEntry]:
        """ρ := random shuffle of every undecided leaf in every non-determined
        tree. A tree is determined when α carries a 1-mark on it (canonical
        single-1 form); every other tree contributes its still-⊥ leaves —
        whether or not α has any 0-marks on it. This guarantees ρ covers all
        undecided leaves so that draining ρ forces every tree to be decided.
        """
        n_leaves = self.irf.n_leaves_per_tree()
        out: List[RhoEntry] = []
        for t in range(self.irf.n_trees):
            m = alpha.decided.get(t, {})
            if any(val == 1 for val in m.values()):
                continue
            for v in range(n_leaves[t]):
                if v not in m:
                    out.append((t, v))
        self.rng.shuffle(out)
        return out

    def _prune_rho(self) -> None:
        """Drop ρ entries that are now decided in self.alpha — explicitly
        (entry in α.decided) or implicitly (canonical dominance: the tree has
        a 1-mark so every leaf in it is fixed).
        """
        determined_trees = {t for t, m in self.alpha.decided.items()
                            if any(val == 1 for val in m.values())}
        self.rho = [(t, v) for (t, v) in self.rho
                    if t not in determined_trees
                    and self.alpha.value(t, v) is None]

    def _vote(self, alpha: PartialAssignment) -> int:
        """Slackless Def-3 vote: count per-tree forced labels, return the
        winner with lex-first tie-break (smallest label index wins ties).
        """
        leaves_per_tree = self.irf.leaves_per_tree()
        counts = [0] * self.irf.n_labels
        for t in range(self.irf.n_trees):
            m = alpha.decided.get(t, {})
            for v, val in m.items():
                if val == 1:
                    counts[leaves_per_tree[t][v]["label_idx"]] += 1
                    break
        # arg-max with lex-first tie-break (smaller label_idx wins ties)
        return max(range(self.irf.n_labels), key=lambda c: (counts[c], -c))
