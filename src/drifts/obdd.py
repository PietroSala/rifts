"""OBDD layer over cell propositional variables.

Implements §2.4 (cell variables), §2.5 (leaf formula Ψ(v)) and §2.10 (cell
exclusivity per feature) from `docs/majority_check_automata_obdd_milp.md`.

Bigger pieces (the running OBDD D from α, the derived function δ, the
absorbing checks) come in subsequent sub-steps; this module is the
self-contained foundation they will compose with.

We use CUDD via the `dd` package (`dd.cudd`). The cell-variable naming is
`c_{feature_idx}_{cell_idx}`. Variables are declared in feature-index order,
then by cell index, so consecutive cells of the same feature sit next to
each other in the BDD variable order — friendly to the per-feature
exactly-one constraints.
"""
from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The compat shim (collections.abc) is only needed when other RFxpl-flavoured
# code is imported in the same process. dd.cudd is self-contained.
import dd.cudd as _cudd

from .indexed_forest import IndexedNode, IndexedRandomForest


# Per-feature contiguous cell range: (lo, hi) inclusive, both in {0, …, K_i}.
CellRange = Tuple[int, int]


# ============================================================================
# Cell-variable encodings (Strategy pattern — see docs/obdd_encodings.md)
# ============================================================================


class CellEncoding:
    """Interface for encoding strategies. Subclasses must declare the cell
    variables in `bdd`, build per-feature range predicates / validity checks,
    and convert a (per-feature) cell choice to a complete variable assignment.
    """
    name: str

    def __init__(self, irf: IndexedRandomForest):
        self.irf = irf

    def n_variables(self) -> int:
        raise NotImplementedError

    def variable_names(self) -> List[str]:
        raise NotImplementedError

    def cells_in_range(self, bdd: _cudd.BDD, feature_idx: int,
                       lo: int, hi: int) -> _cudd.Function:
        """BDD for 'feature `feature_idx`'s cell index lies in [lo, hi]'."""
        raise NotImplementedError

    def feature_validity(self, bdd: _cudd.BDD, feature_idx: int) -> _cudd.Function:
        """BDD for 'this feature's variables encode some valid cell'.

        For one-hot this is exactly-one; for binary it is the validity
        predicate `J < M_i` (or TRUE when `M_i` is a power of 2).
        """
        raise NotImplementedError

    def assignment_from_cells(self,
                              cells: Sequence[int]) -> Dict[str, bool]:
        """Per-feature cell choice → complete variable assignment for
        `bdd.let(...)`."""
        raise NotImplementedError


# ----- A. one-hot ------------------------------------------------------------


class OneHotEncoding(CellEncoding):
    """One Boolean variable `c_{i,j}` per (feature, cell)."""
    name = "one_hot"

    def __init__(self, irf: IndexedRandomForest):
        super().__init__(irf)
        # var-name layout decided at declare()
        self._names: List[str] = []
        for i in range(irf.n_features):
            for j in range(self._M(i)):
                self._names.append(f"c_{i}_{j}")

    def _M(self, i: int) -> int:
        return len(self.irf.EU[i]) + 1

    def n_variables(self) -> int:
        return len(self._names)

    def variable_names(self) -> List[str]:
        return self._names

    def cells_in_range(self, bdd, feature_idx, lo, hi):
        out = bdd.false
        for j in range(lo, hi + 1):
            out = out | bdd.var(f"c_{feature_idx}_{j}")
        return out

    def feature_validity(self, bdd, feature_idx):
        M = self._M(feature_idx)
        vars_ = [bdd.var(f"c_{feature_idx}_{j}") for j in range(M)]
        # at-least-one
        at_least = bdd.false
        for v in vars_:
            at_least = at_least | v
        # at-most-one
        at_most = bdd.true
        for j in range(len(vars_)):
            for k in range(j + 1, len(vars_)):
                at_most = at_most & ~(vars_[j] & vars_[k])
        return at_least & at_most

    def assignment_from_cells(self, cells):
        sub: Dict[str, bool] = {}
        for i, j_chosen in enumerate(cells):
            M = self._M(i)
            for j in range(M):
                sub[f"c_{i}_{j}"] = (j == j_chosen)
        return sub


# ----- B. binary -------------------------------------------------------------


class BinaryEncoding(CellEncoding):
    """`b_i = ⌈log₂(M_i)⌉` Boolean variables `b_{i,k}` per feature, interpreted
    as the cell index in binary (LSB = bit 0). Features with `M_i = 1` get
    zero bits (the only valid cell is 0, no variables needed).
    """
    name = "binary"

    def __init__(self, irf: IndexedRandomForest):
        super().__init__(irf)
        self._bits_per_feature: List[int] = []
        self._names: List[str] = []
        for i in range(irf.n_features):
            M = self._M(i)
            b = 0 if M <= 1 else math.ceil(math.log2(M))
            self._bits_per_feature.append(b)
            for k in range(b):
                self._names.append(f"b_{i}_{k}")

    def _M(self, i: int) -> int:
        return len(self.irf.EU[i]) + 1

    def n_bits(self, feature_idx: int) -> int:
        return self._bits_per_feature[feature_idx]

    def n_variables(self) -> int:
        return len(self._names)

    def variable_names(self) -> List[str]:
        return self._names

    def _exact_match(self, bdd, feature_idx, j: int) -> _cudd.Function:
        """BDD for 'feature `feature_idx`'s bits encode integer j'."""
        b = self._bits_per_feature[feature_idx]
        if b == 0:
            return bdd.true if j == 0 else bdd.false
        out = bdd.true
        for k in range(b):
            bit_var = bdd.var(f"b_{feature_idx}_{k}")
            if (j >> k) & 1:
                out = out & bit_var
            else:
                out = out & ~bit_var
        return out

    def cells_in_range(self, bdd, feature_idx, lo, hi):
        """Build BDD for 'cell index ∈ [lo, hi]' as a range-comparator: an
        O(b)-size BDD via two recursive constructions (`bits ≥ lo`,
        `bits ≤ hi`) ANDed together. See docs/obdd_encodings.md §C.
        """
        b = self._bits_per_feature[feature_idx]
        if b == 0:
            # only cell 0 exists
            return bdd.true if (lo <= 0 <= hi) else bdd.false
        if lo > hi:
            return bdd.false
        max_int = (1 << b) - 1
        ge_bdd = bdd.true if lo <= 0       else self._ge_rec(bdd, feature_idx, b - 1, lo)
        le_bdd = bdd.true if hi >= max_int else self._le_rec(bdd, feature_idx, b - 1, hi)
        return ge_bdd & le_bdd

    def _ge_rec(self, bdd, feat, k, residual):
        """BDD for 'integer encoded by bits[k..0] ≥ residual'. O(b) ITE nodes."""
        if residual <= 0:
            return bdd.true
        if k < 0:
            return bdd.false           # empty bits encode 0 < residual
        pow_k = 1 << k
        lo_k = (residual >> k) & 1
        bit_k = bdd.var(f"b_{feat}_{k}")
        if lo_k == 1:
            # residual ≥ pow_k. bit_k=0 ⇒ FALSE; bit_k=1 ⇒ recurse on residual-pow_k.
            return bdd.apply("ite", bit_k,
                             self._ge_rec(bdd, feat, k - 1, residual - pow_k),
                             bdd.false)
        # lo_k = 0: residual < pow_k. bit_k=1 ⇒ TRUE; bit_k=0 ⇒ recurse on residual.
        return bdd.apply("ite", bit_k,
                         bdd.true,
                         self._ge_rec(bdd, feat, k - 1, residual))

    def _le_rec(self, bdd, feat, k, residual):
        """BDD for 'integer encoded by bits[k..0] ≤ residual'. O(b) ITE nodes."""
        if residual < 0:
            return bdd.false
        if k < 0:
            return bdd.true            # empty bits encode 0 ≤ residual
        pow_k = 1 << k
        hi_k = (residual >> k) & 1
        bit_k = bdd.var(f"b_{feat}_{k}")
        if hi_k == 1:
            # residual ≥ pow_k. bit_k=0 ⇒ TRUE; bit_k=1 ⇒ recurse on residual-pow_k.
            return bdd.apply("ite", bit_k,
                             self._le_rec(bdd, feat, k - 1, residual - pow_k),
                             bdd.true)
        # hi_k = 0: residual < pow_k. bit_k=1 ⇒ FALSE; bit_k=0 ⇒ recurse on residual.
        return bdd.apply("ite", bit_k,
                         bdd.false,
                         self._le_rec(bdd, feat, k - 1, residual))

    def feature_validity(self, bdd, feature_idx):
        M = self._M(feature_idx)
        b = self._bits_per_feature[feature_idx]
        if b == 0 or M == (1 << b):
            # perfect power of 2 or single cell ⇒ every bit assignment is
            # automatically a valid cell; no constraint to add.
            return bdd.true
        # 'bits encode integer < M' = range-comparator over [0, M-1]
        return self.cells_in_range(bdd, feature_idx, 0, M - 1)

    def assignment_from_cells(self, cells):
        sub: Dict[str, bool] = {}
        for i, j_chosen in enumerate(cells):
            b = self._bits_per_feature[i]
            for k in range(b):
                sub[f"b_{i}_{k}"] = bool((j_chosen >> k) & 1)
        return sub


# ============================================================================


@dataclass
class BootstrapConfig:
    """How a worker should populate its OBDDContext.

    - mode="per_worker"  — build everything locally (no Redis I/O). Cheap,
      each worker pays the build cost independently.
    - mode="shared"       — coordinate via Redis:
        * if `<ds>:OBDD:ready == "1"`, load the DDDMP dump.
        * else acquire `<ds>:OBDD:builder_lock` (SET NX EX), build locally,
          dump to Redis, release.
        * if the lock is taken, poll until ready or timeout.
    """
    mode: str = "per_worker"            # "per_worker" | "shared"
    redis: object | None = None          # redis.Redis client (no decode_responses)
    dataset: Optional[str] = None
    builder_lock_ttl: int = 600
    poll_interval: float = 1.0
    poll_timeout: float = 1800.0


@dataclass
class OBDDContext:
    """OBDD over the cell variables of one IndexedRandomForest.

    Construction declares every cell variable for the chosen encoding (see
    `docs/obdd_encodings.md`). After construction, call `bootstrap(config)`
    to populate `leaf_formulas` and `cell_excl` — either locally (cheap,
    per-worker) or via a shared Redis-backed JSON dump.
    """
    irf: IndexedRandomForest
    bdd: _cudd.BDD
    enc: CellEncoding
    # cached per-feature validity / exclusivity factors so we don't rebuild them.
    _feat_validity: Dict[int, "_cudd.Function"]
    # populated by bootstrap()
    leaf_formulas: List[Dict[int, "_cudd.Function"]] = field(default_factory=list)
    cell_excl: Optional["_cudd.Function"] = None
    is_bootstrapped: bool = False

    # ----- construction ----------------------------------------------------

    @classmethod
    def for_forest(cls, irf: IndexedRandomForest,
                   encoding: str = "one_hot",
                   cudd_cache_size: int = 1_000_000) -> "OBDDContext":
        bdd = _cudd.BDD()
        try:
            bdd.configure(max_cache_hard=cudd_cache_size)
        except Exception:
            # Older `dd` versions raise on unknown config keys; not fatal.
            pass
        if encoding == "one_hot":
            enc: CellEncoding = OneHotEncoding(irf)
        elif encoding == "binary":
            enc = BinaryEncoding(irf)
        else:
            raise ValueError(f"unknown encoding {encoding!r}; "
                             f"expected 'one_hot' or 'binary'")
        names = enc.variable_names()
        if names:
            bdd.declare(*names)
        return cls(irf=irf, bdd=bdd, enc=enc, _feat_validity={})

    # ----- shape helpers ---------------------------------------------------

    def n_cells_for(self, feature_idx: int) -> int:
        """K_i + 1 — the number of cells for feature `feature_idx`."""
        return len(self.irf.EU[feature_idx]) + 1

    def cell_index_of_value(self, feature_idx: int, value: float) -> int:
        """Return the cell index j s.t. `value` falls in cell (r_{j-1}, r_j]
        (with r_{-1} = -∞, r_K = +∞). This is the `e_pos` of the trivial-ICF.
        """
        eu = self.irf.EU[feature_idx]
        # smallest p with value <= EU[p]; or |EU| if none
        lo, hi, ans = 0, len(eu) - 1, len(eu)
        while lo <= hi:
            mid = (lo + hi) // 2
            if value <= float(eu[mid]):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

    # ----- §2.5  Leaf formula Ψ(v) ----------------------------------------

    def leaf_formula(self, ranges: Dict[int, CellRange]):
        """Build Ψ(v) given the per-feature cell-index range `(lo, hi)` allowed
        by v's root-to-leaf path. Features whose range is the full
        `{0, …, K_i}` are unconstrained and skipped. Cell-encoding-specific
        construction is delegated to `self.enc.cells_in_range`.
        """
        psi = self.bdd.true
        for i, (lo, hi) in ranges.items():
            K = self.n_cells_for(i) - 1
            if lo == 0 and hi == K:
                continue
            psi = psi & self.enc.cells_in_range(self.bdd, i, lo, hi)
        return psi

    def compute_leaf_formulas(
        self,
        on_leaf: "Optional[Any]" = None,
    ) -> List[Dict[int, "_cudd.Function"]]:
        """Walk every tree, produce `{leaf_idx: Ψ(v)}` per tree by accumulating
        the cell-range constraint along the root-to-leaf path.

        `on_leaf` (optional, callable) is invoked once per leaf after its
        formula has been emitted. Used by drivers to drive a tqdm progress
        bar — see `compute_leaf_formulas(on_leaf=pbar.update)`.
        """
        out: List[Dict[int, _cudd.Function]] = []
        for tree in self.irf.trees:
            tree_map: Dict[int, _cudd.Function] = {}
            ranges = {i: (0, self.n_cells_for(i) - 1)
                      for i in range(self.irf.n_features)}
            self._descend(tree, ranges, tree_map, on_leaf)
            out.append(tree_map)
        return out

    def _descend(self, node: IndexedNode,
                 ranges: Dict[int, CellRange],
                 out: Dict[int, "_cudd.Function"],
                 on_leaf: "Optional[Any]" = None) -> None:
        if node["type"] == "leaf":
            out[node["leaf_idx"]] = self.leaf_formula(ranges)
            if on_leaf is not None:
                on_leaf()
            return
        i = node["feature_idx"]
        t = node["threshold_pos"]
        lo, hi = ranges[i]
        # Left branch: tree test `x ≤ EU[i][t]` → cells j with j ≤ t.
        new_hi = min(hi, t)
        if lo <= new_hi:
            ranges[i] = (lo, new_hi)
            self._descend(node["low"], ranges, out, on_leaf)
            ranges[i] = (lo, hi)
        # Right branch: `x > EU[i][t]` → cells j with j ≥ t + 1.
        new_lo = max(lo, t + 1)
        if new_lo <= hi:
            ranges[i] = (new_lo, hi)
            self._descend(node["high"], ranges, out, on_leaf)
            ranges[i] = (lo, hi)

    # ----- §2.10  Per-feature constraint (exactly-one for A, validity for B) ----

    def cell_exclusivity_for(self, feature_idx: int) -> "_cudd.Function":
        """Per-feature validity / exclusivity constraint, delegated to the
        encoding (exactly-one for one-hot; J < M_i for binary or TRUE when
        M_i is a power of 2). Cached.
        """
        cached = self._feat_validity.get(feature_idx)
        if cached is not None:
            return cached
        out = self.enc.feature_validity(self.bdd, feature_idx)
        self._feat_validity[feature_idx] = out
        return out

    def cell_exclusivity(self) -> "_cudd.Function":
        """⋀_i per-feature validity constraint over the whole forest."""
        out = self.bdd.true
        for i in range(self.irf.n_features):
            out = out & self.cell_exclusivity_for(i)
        return out

    # ----- §2.10  Cell-tuple → variable assignment ------------------------

    def assignment_from_cells(self,
                              cell_of_feature: Sequence[int]) -> Dict[str, bool]:
        """Build a complete variable assignment from a per-feature cell choice.
        Encoding-specific (one-hot vs binary)."""
        return self.enc.assignment_from_cells(cell_of_feature)

    def sample_cell_tuple(self, x: Sequence[float]) -> List[int]:
        """Return `[cell_index_of_value(i, x[i]) for i in 0..n_features-1]`."""
        return [self.cell_index_of_value(i, float(x[i]))
                for i in range(self.irf.n_features)]

    # ----- §2.9  Running OBDD D and §2.8 Derived δ ------------------------

    def build_D(self, alpha) -> "_cudd.Function":
        """The running OBDD with cell-exclusivity pre-conjoined:

            D = cell_excl  ∧  ⋀_{t, v ∈ dom(α_t)}  cond(α_t(v), Ψ(v))

        with `cond(0, Ψ) = ¬Ψ` and `cond(1, Ψ) = Ψ`. Pre-conjoining cell
        exclusivity makes `is_corridor_unsat(D)` directly test whether any
        real sample satisfies α (the equivalent of §2.10's D').
        """
        if not self.is_bootstrapped:
            raise RuntimeError(
                "OBDDContext not bootstrapped — call bootstrap() first"
            )
        D = self.cell_excl
        for tree_idx, leaf_map in alpha.decided.items():
            for leaf_idx, val in leaf_map.items():
                psi = self.leaf_formulas[tree_idx][leaf_idx]
                D = D & (psi if val == 1 else ~psi)
        return D

    def conjoin_D(self, D: "_cudd.Function", tree_idx: int, leaf_idx: int,
                  val: int) -> "_cudd.Function":
        """Incremental update: `D' = D ∧ cond(val, Ψ(tree, leaf))`. Used by
        Step to add one freshly-committed leaf without rebuilding D."""
        if val not in (0, 1):
            raise ValueError("val must be 0 or 1")
        psi = self.leaf_formulas[tree_idx][leaf_idx]
        return D & (psi if val == 1 else ~psi)

    def is_corridor_unsat(self, D: "_cudd.Function") -> bool:
        """True iff `D ≡ ⊥`, i.e., no real sample satisfies α — α is faulty
        and the corridor is structurally empty (`absorbing = 0`)."""
        return D == self.bdd.false

    def compute_delta(self, alpha, D: "_cudd.Function"):
        """Derived function δ (§2.8): for every leaf `v` with `α_t(v) = ⊥`,

            δ_t(v) = 0   if   D ∧ Ψ(v)  ≡ ⊥
            δ_t(v) = 1   if   D ∧ ¬Ψ(v) ≡ ⊥
            δ_t(v) = ⊥   otherwise.

        Returns a fresh `PartialAssignment` containing only the *derived*
        entries (callers combine with `α` to get the full constraint state).
        """
        if not self.is_bootstrapped:
            raise RuntimeError(
                "OBDDContext not bootstrapped — call bootstrap() first"
            )
        from .partial_assignment import PartialAssignment   # local import to avoid cycle
        delta = PartialAssignment()
        bdd = self.bdd
        FALSE = bdd.false
        for tree_idx, tree_map in enumerate(self.leaf_formulas):
            alpha_t = alpha.decided.get(tree_idx, {})
            for leaf_idx, psi in tree_map.items():
                if leaf_idx in alpha_t:
                    continue
                if (D & psi) == FALSE:
                    delta.set(tree_idx, leaf_idx, 0)
                elif (D & ~psi) == FALSE:
                    delta.set(tree_idx, leaf_idx, 1)
        return delta

    # ----- Bootstrap (per-worker OR shared via Redis DDDMP) ---------------

    def bootstrap(self, config: BootstrapConfig | None = None,
                  on_leaf: "Optional[Any]" = None) -> None:
        """Populate `leaf_formulas` and `cell_excl` per the chosen mode.

        Idempotent: returns immediately if already bootstrapped. The
        `on_leaf` callback is forwarded to `_build_local` and called once
        per Ψ(v) emitted (drivers wire it to tqdm). Ignored on the
        warm-load path (which doesn't go leaf-by-leaf).
        """
        if self.is_bootstrapped:
            return
        cfg = config or BootstrapConfig()
        if cfg.mode == "per_worker":
            self._build_local(on_leaf=on_leaf)
            return
        if cfg.mode != "shared":
            raise ValueError(f"unknown bootstrap mode {cfg.mode!r}")

        if cfg.redis is None or not cfg.dataset:
            raise ValueError("shared bootstrap requires `redis` and `dataset`")

        ds = cfg.dataset
        r = cfg.redis
        ready_key = f"{ds}:OBDD:ready"
        lock_key = f"{ds}:OBDD:builder_lock"

        # 1) Fast path: ready blob exists, just load.
        if r.get(ready_key) in (b"1", "1"):
            self._load_from_redis(r, ds)
            return

        # 2) Try to become the builder. SET NX EX — atomic.
        worker_id = f"{os.getpid()}.{int(time.time()*1000)}"
        got_lock = r.set(lock_key, worker_id, nx=True, ex=cfg.builder_lock_ttl)
        if got_lock:
            try:
                self._build_local(on_leaf=on_leaf)
                self._dump_to_redis(r, ds)
            finally:
                # Only delete the lock if we still own it (best-effort).
                cur = r.get(lock_key)
                if cur in (worker_id, worker_id.encode()):
                    r.delete(lock_key)
            return

        # 3) Someone else is building; poll for readiness.
        t0 = time.time()
        while time.time() - t0 < cfg.poll_timeout:
            if r.get(ready_key) in (b"1", "1"):
                self._load_from_redis(r, ds)
                return
            time.sleep(cfg.poll_interval)
        raise TimeoutError(
            f"shared bootstrap for {ds!r}: build did not complete within "
            f"{cfg.poll_timeout}s"
        )

    def _build_local(self, on_leaf: "Optional[Any]" = None) -> None:
        """The expensive build path. Computes every Ψ(v) and every per-feature
        cell-exclusivity factor; sets `leaf_formulas`, `cell_excl`,
        `is_bootstrapped`. `on_leaf` is forwarded to `compute_leaf_formulas`.
        """
        self.leaf_formulas = self.compute_leaf_formulas(on_leaf=on_leaf)
        for i in range(self.irf.n_features):
            self.cell_exclusivity_for(i)         # populates the cache
        self.cell_excl = self.cell_exclusivity()
        self.is_bootstrapped = True

    # ---- DDDMP I/O via Redis (binary base64-encoded blob) ---------------

    def _ds_keys(self, ds: str) -> Dict[str, str]:
        e = self.enc.name
        return {
            "blob":   f"{ds}:OBDD:{e}:blob",
            "meta":   f"{ds}:OBDD:{e}:meta",
            "ready":  f"{ds}:OBDD:{e}:ready",
            "lock":   f"{ds}:OBDD:{e}:builder_lock",
        }

    def _canonical_roots(self) -> tuple[list["_cudd.Function"], dict]:
        """Return (roots_list, root_layout) in a deterministic order:
        - first: every Ψ(v) for (tree 0, leaves in leaf_idx order), then tree 1, …
        - then: every per-feature exclusivity factor, feature 0..n_features-1.
        `root_layout` records the offsets/lengths so the loader can slice.
        """
        roots: list[_cudd.Function] = []
        psi_offset = 0
        leaves_per_tree = [len(m) for m in self.leaf_formulas]
        for tj, tree_map in enumerate(self.leaf_formulas):
            for lv in sorted(tree_map):
                roots.append(tree_map[lv])
        excl_offset = len(roots)
        for i in range(self.irf.n_features):
            roots.append(self.cell_exclusivity_for(i))
        layout = {
            "psi_offset": psi_offset,
            "leaves_per_tree": leaves_per_tree,
            "excl_offset": excl_offset,
            "n_features": self.irf.n_features,
        }
        return roots, layout

    def _dump_to_redis(self, r, ds: str) -> None:
        """Dump every Ψ(v) and the per-feature cell-exclusivity factors via
        the JSON BDD format into Redis. Idempotent: writes meta + blob + ready=1.
        """
        roots, layout = self._canonical_roots()
        # dd's JSON dump creates a relative `__shelve__/` next to cwd and
        # raises if it already exists; cd into a scratch dir to keep this
        # local and idempotent.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work:
            os.chdir(work)
            try:
                path = os.path.join(work, "dump.json")
                self.bdd.dump(path, roots=roots, filetype="json")
                with open(path, "rb") as fh:
                    blob = fh.read()
            finally:
                os.chdir(cwd)

        meta = {
            "dataset": ds,
            "n_features": self.irf.n_features,
            "n_trees": self.irf.n_trees,
            "leaves_per_tree": self.irf.n_leaves_per_tree(),
            "encoding": self.enc.name,       # see docs/obdd_encodings.md
            "layout": layout,
        }
        keys = self._ds_keys(ds)
        r.set(keys["blob"], base64.b64encode(blob).decode("ascii"))
        r.set(keys["meta"], json.dumps(meta))
        r.set(keys["ready"], "1")

    def _load_from_redis(self, r, ds: str) -> None:
        """Load the JSON BDD blob from Redis and rehydrate `leaf_formulas` +
        per-feature cell exclusivity from the dumped roots (in canonical order)."""
        keys = self._ds_keys(ds)
        b64_raw = r.get(keys["blob"])
        meta_raw = r.get(keys["meta"])
        if b64_raw is None or meta_raw is None:
            raise RuntimeError(f"shared bootstrap for {ds!r}: blob or meta missing")
        if isinstance(b64_raw, bytes):
            b64_raw = b64_raw.decode("ascii")
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        meta = json.loads(meta_raw)
        if (meta["n_features"] != self.irf.n_features
                or meta["n_trees"] != self.irf.n_trees):
            raise RuntimeError(
                f"OBDD dump meta does not match IRF for {ds!r}: "
                f"meta={meta!r}, irf=({self.irf.n_features}, {self.irf.n_trees})"
            )
        if meta.get("encoding", "one_hot") != self.enc.name:
            raise RuntimeError(
                f"OBDD dump for {ds!r} uses encoding {meta.get('encoding')!r}, "
                f"but this context expects {self.enc.name!r} "
                f"(see docs/obdd_encodings.md)"
            )

        # cd into a scratch dir for the same `__shelve__` reason on load.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as work:
            os.chdir(work)
            try:
                path = os.path.join(work, "dump.json")
                with open(path, "wb") as fh:
                    fh.write(base64.b64decode(b64_raw))
                roots = self.bdd.load(path)
            finally:
                os.chdir(cwd)

        layout = meta["layout"]
        leaves_per_tree = layout["leaves_per_tree"]
        # rehydrate Ψ in canonical order
        self.leaf_formulas = [dict() for _ in range(meta["n_trees"])]
        idx = layout["psi_offset"]
        for tj in range(meta["n_trees"]):
            for lv in range(leaves_per_tree[tj]):
                self.leaf_formulas[tj][lv] = roots[idx]
                idx += 1
        # rehydrate per-feature exclusivity
        idx = layout["excl_offset"]
        for i in range(self.irf.n_features):
            self._cell_excl_per_feat[i] = roots[idx]
            idx += 1
        # full cell_excl is the AND of the cached per-feature factors
        self.cell_excl = self.cell_exclusivity()
        self.is_bootstrapped = True
