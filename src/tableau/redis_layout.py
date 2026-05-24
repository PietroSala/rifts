"""Redis key layout for the tableau search.

All tableau-layer state for one ``(dataset, sample)`` lives under
``<ds>:sample:<k>:tableau:…`` per ``code/docs/tableau_outline.md`` §15.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TableauKeys:
    """Key constructor for the tableau on a single ``(dataset, sample_idx)``."""
    dataset: str
    sample_idx: int

    @property
    def prefix(self) -> str:
        return f"{self.dataset}:sample:{self.sample_idx}:tableau"

    @property
    def state(self) -> str:        return f"{self.prefix}:state"
    @property
    def leaves(self) -> str:       return f"{self.prefix}:leaves"
    @property
    def inflight(self) -> str:     return f"{self.prefix}:inflight"
    @property
    def internals(self) -> str:    return f"{self.prefix}:internals"
    @property
    def workers(self) -> str:      return f"{self.prefix}:workers"
    @property
    def best(self) -> str:         return f"{self.prefix}:best"

    def node(self, node_id: str) -> str:
        return f"{self.prefix}:node:{node_id}"

    @property
    def claim_prefix(self) -> str:
        # Lua scripts use this as a prefix; the script appends node_id.
        return f"{self.prefix}:claim:"

    def claim(self, node_id: str) -> str:
        return f"{self.claim_prefix}{node_id}"
