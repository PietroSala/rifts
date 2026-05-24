"""Strict (cell-consistent) majority check via CEGAR.

Adds a complete alternative to the conservative ``majority_check`` (Adv_max
closed form). The conservative oracle treats each free tree as if it could
*independently* pick any leaf in ``L_t``; in reality the choice of a leaf in
one tree may impose cell constraints that another tree cannot satisfy
simultaneously. CEGAR closes that gap:

  1. **MILP**: for each competitor ``c ≠ c⋆``, solve a per-tree leaf-selection
     MILP maximising ``votes(c) − votes(c⋆)`` subject to per-tree
     exactly-one, profile membership, and any currently-known no-good
     constraints.
  2. **Realisability**: if the MILP returns an assignment that violates the
     lex-first Def-3 threshold, check via the OBDD whether a real sample
     actually realises that leaf-tuple (``D ∧ ⋀_t Ψ(v_t) ≢ ⊥``).
  3. **Minimal no-good**: if not realisable, deletion-shrink the
     leaf-tuple to a minimal infeasible subset (MUS-style) and add it as
     a no-good cut. Loop.

Termination: each iteration adds at least one new no-good, the no-good
lattice is finite, and infeasibility eventually drops the per-competitor
``Adv_max`` below the threshold.

A **global infeasibility cache** is maintained on the instance — these
``frozenset[(tree, leaf)]`` cuts hold across ICFs because they originate
from forest-level cell-exclusivity. The cache is filtered to "relevant"
(profile-membership) cuts before each MILP call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds

from .indexed_forest import IndexedRandomForest
from .obdd import OBDDContext
from .partial_assignment import PartialAssignment
from .profile import forest_profile


LeafTuple = Tuple[Tuple[int, int], ...]   # ((tree, leaf), …) sorted


@dataclass
class CEGARResult:
    verdict: str                                  # "good" | "bad"
    witness: Optional[PartialAssignment]          # set on "bad"
    n_milp_calls: int = 0
    n_no_goods_used: int = 0
    n_no_goods_learned: int = 0
    elapsed_s: float = 0.0


@dataclass
class CEGARMajority:
    """Strict majority check via CEGAR over the conservative Adv_max.

    Construction:
      ``CEGARMajority(irf, ctx)`` where ``ctx`` is a bootstrapped
      ``OBDDContext`` (the BDD over leaf cell-formulas).

    Call:
      ``is_good_strict(icf, c_star) -> CEGARResult``.
    """
    irf: IndexedRandomForest
    ctx: OBDDContext

    # Global infeasibility cache: tuples of (tree_idx, leaf_idx) that cannot
    # co-occur in any real sample (forest-level cell-exclusivity).
    infeasibility_cache: Set[FrozenSet[Tuple[int, int]]] = field(default_factory=set)

    # Stats (cumulative across calls)
    total_milp_calls: int = 0
    total_no_goods_learned: int = 0

    # ----------------------------- public API ---------------------------------

    def is_good_strict(self, icf, c_star: int) -> CEGARResult:
        """Return the strict (cell-consistent) Good/Bad verdict for the ICF."""
        import time
        t0 = time.time()
        profile = forest_profile(self.irf, icf)
        leaves_per_tree = self.irf.leaves_per_tree()
        n_labels = self.irf.n_labels

        n_milp = 0
        n_used = 0
        n_learned = 0

        for c in range(n_labels):
            if c == c_star:
                continue
            res = self._search_bad_for_competitor(
                profile, leaves_per_tree, c, c_star,
            )
            n_milp   += res["n_milp"]
            n_used   += res["n_used"]
            n_learned += res["n_learned"]
            if res["witness"] is not None:
                return CEGARResult(
                    verdict="bad", witness=res["witness"],
                    n_milp_calls=n_milp,
                    n_no_goods_used=n_used,
                    n_no_goods_learned=n_learned,
                    elapsed_s=time.time() - t0,
                )

        return CEGARResult(
            verdict="good", witness=None,
            n_milp_calls=n_milp,
            n_no_goods_used=n_used,
            n_no_goods_learned=n_learned,
            elapsed_s=time.time() - t0,
        )

    # ----------------------------- core loop ----------------------------------

    def _search_bad_for_competitor(
        self,
        profile: Sequence[Set[int]],
        leaves_per_tree,
        c: int,
        c_star: int,
    ) -> dict:
        """For one competitor ``c``, CEGAR-loop to either find a
        cell-consistent Bad witness, or prove none exists.

        Returns ``{"witness", "n_milp", "n_used", "n_learned"}``."""
        # Pre-filter the global cache to no-goods that are a subset of this
        # ICF's profile (the only cuts that can affect this MILP).
        relevant: List[FrozenSet[Tuple[int, int]]] = [
            F for F in self.infeasibility_cache
            if all(v in profile[t] for (t, v) in F)
        ]
        n_used_global = len(relevant)
        n_milp = 0
        n_learned = 0

        while True:
            n_milp += 1
            self.total_milp_calls += 1
            sol = self._milp_max_adv(profile, leaves_per_tree, c, c_star,
                                     relevant)
            if sol is None:
                # MILP infeasible — no leaf-tuple satisfies the constraints
                # (e.g. ruled out by no-goods). c cannot beat c⋆.
                return {"witness": None, "n_milp": n_milp,
                        "n_used": n_used_global, "n_learned": n_learned}
            adv, witness_tuple = sol
            if not self._violates_threshold(adv, c, c_star):
                return {"witness": None, "n_milp": n_milp,
                        "n_used": n_used_global, "n_learned": n_learned}

            if self._is_realisable(witness_tuple):
                # Real Bad witness.
                alpha = self._tuple_to_partial(witness_tuple)
                return {"witness": alpha, "n_milp": n_milp,
                        "n_used": n_used_global, "n_learned": n_learned}

            # Spurious witness — shrink to a minimal infeasibility, cache,
            # forbid and retry.
            F = self._minimal_infeasibility(witness_tuple)
            if F in self.infeasibility_cache:
                # Should not happen if filtering is consistent — defensive.
                relevant.append(F)
            else:
                self.infeasibility_cache.add(F)
                self.total_no_goods_learned += 1
                n_learned += 1
                relevant.append(F)

    # ----------------------------- MILP ---------------------------------------

    def _milp_max_adv(
        self,
        profile: Sequence[Set[int]],
        leaves_per_tree,
        c: int,
        c_star: int,
        forbidden: List[FrozenSet[Tuple[int, int]]],
    ):
        """Per-c MILP. Variables y_{t,v} ∈ {0,1} for v ∈ profile[t].

        Constraints:
          ∑_v y_{t,v} = 1                       ∀t
          ∑_{(t,v) ∈ F} y_{t,v} ≤ |F|−1         ∀F ∈ forbidden

        Objective (minimised by milp; we negate the advantage):
          minimise −(votes(c) − votes(c_star))
          = minimise ∑_{t,v} y_{t,v} · (1[label = c_star] − 1[label = c])

        Returns ``(adv, leaf_tuple)`` or ``None`` if infeasible.
        """
        n_trees = self.irf.n_trees
        # Flatten variable layout
        var_index: Dict[Tuple[int, int], int] = {}
        var_label: Dict[int, int] = {}              # var_idx → label_idx
        coef = []                                    # objective coefficients
        for t in range(n_trees):
            for v in sorted(profile[t]):
                vi = len(var_index)
                var_index[(t, v)] = vi
                lab = leaves_per_tree[t][v]["label_idx"]
                var_label[vi] = lab
                # minimise −advantage ⇒ negative coef for c-label, positive for c_star
                if lab == c:
                    coef.append(-1.0)
                elif lab == c_star:
                    coef.append(+1.0)
                else:
                    coef.append(0.0)
        n_vars = len(var_index)
        if n_vars == 0:
            return None

        # per-tree exactly-one
        A_eq_rows = []
        b_eq_lo = []
        b_eq_hi = []
        for t in range(n_trees):
            row = np.zeros(n_vars, dtype=float)
            empty = True
            for v in profile[t]:
                row[var_index[(t, v)]] = 1.0
                empty = False
            if empty:
                # Tree with empty profile — no consistent leaf selection exists.
                return None
            A_eq_rows.append(row)
            b_eq_lo.append(1.0); b_eq_hi.append(1.0)

        # no-good constraints
        A_le_rows = []; b_le_lo = []; b_le_hi = []
        for F in forbidden:
            row = np.zeros(n_vars, dtype=float)
            for (t, v) in F:
                if (t, v) in var_index:
                    row[var_index[(t, v)]] = 1.0
            A_le_rows.append(row)
            b_le_lo.append(-np.inf); b_le_hi.append(float(len(F)) - 1.0)

        A_all = np.vstack(A_eq_rows + A_le_rows) if A_le_rows else np.vstack(A_eq_rows)
        b_lo = b_eq_lo + b_le_lo
        b_hi = b_eq_hi + b_le_hi
        constraints = LinearConstraint(A_all, b_lo, b_hi)

        integrality = np.ones(n_vars, dtype=int)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

        res = milp(
            c=np.array(coef, dtype=float),
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
        )
        if not res.success or res.x is None:
            return None

        # Recover leaf-tuple
        x = res.x
        picks: List[Tuple[int, int]] = []
        # Quick scan
        for (t, v), vi in var_index.items():
            if x[vi] > 0.5:
                picks.append((t, v))
        picks.sort()
        adv = int(round(-res.fun))   # advantage value
        return (adv, tuple(picks))

    # ----------------------------- realisability ------------------------------

    def _is_realisable(self, witness: LeafTuple) -> bool:
        """The OBDD's cell-exclusivity is already pre-conjoined into D; we
        AND in the Ψ(v_t) for each picked leaf and check unsat."""
        D = self.ctx.cell_excl
        bdd = self.ctx.bdd
        for (t, v) in witness:
            psi = self.ctx.leaf_formulas[t][v]
            D = D & psi
            if D == bdd.false:
                return False
        return True

    # ----------------------------- minimal infeasibility ----------------------

    def _minimal_infeasibility(self, witness: LeafTuple) -> FrozenSet[Tuple[int, int]]:
        """Deletion-based MUS extraction. Returns a minimal (subset-minimal)
        infeasible leaf-tuple ⊆ witness."""
        current = list(witness)
        i = 0
        while i < len(current):
            candidate = current[:i] + current[i + 1:]
            if candidate and not self._is_realisable(tuple(candidate)):
                # Still infeasible without this element — drop it.
                current = candidate
            else:
                i += 1
        return frozenset(current)

    # ----------------------------- helpers -----------------------------------

    def _violates_threshold(self, adv: int, c: int, c_star: int) -> bool:
        """Def-3 lex-first violation: c > c_star ⇒ adv > 0; c < c_star ⇒ adv ≥ 0."""
        if c > c_star:
            return adv > 0
        return adv >= 0

    def _tuple_to_partial(self, witness: LeafTuple) -> PartialAssignment:
        n_leaves = self.irf.n_leaves_per_tree()
        alpha = PartialAssignment(n_leaves_per_tree=tuple(n_leaves))
        for (t, v) in witness:
            alpha.set(t, v, 1)
        return alpha


# ----------------------------- top-level convenience -------------------------


def is_good_strict(
    irf: IndexedRandomForest,
    ctx: OBDDContext,
    icf,
    c_star: int,
    *,
    instance: Optional[CEGARMajority] = None,
) -> CEGARResult:
    """Convenience wrapper: build an ephemeral ``CEGARMajority`` (or reuse the
    provided one for cache continuity) and run the strict check."""
    if instance is None:
        instance = CEGARMajority(irf=irf, ctx=ctx)
    return instance.is_good_strict(icf, c_star)
