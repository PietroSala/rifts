"""Centralised path resolution for the rifts-local scripts.

Every script (top-level or under sub-packages such as ``max_iaxp/``) imports
this module to obtain ``STATE_ROOT`` (mirror of the live ``experiments/``
working state, under ``rifts-local/current-state/``) and ``DATA_ROOT`` (the
UCR archive under ``rifts-local/dataset/Univariate_ts/``).

Importing this module also extends ``sys.path`` with the ``src/`` and
``scripts/`` directories of ``rifts-local`` so the ``drifts``, ``tableau``,
``cache`` packages and the cross-script imports (e.g. ``import build_hasse``)
resolve regardless of the current working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _find_rifts_root(start: Path) -> Path:
    """Walk up from ``start`` until we find the rifts-local root.

    The marker is the presence of both ``dataset/`` and ``factory-setting/``
    immediately under the candidate directory.
    """
    p = start.resolve()
    while True:
        if (p / "dataset").is_dir() and (p / "factory-setting").is_dir():
            return p
        if p == p.parent:
            raise RuntimeError(
                "Could not locate the rifts-local root above "
                f"{start!s}: no parent directory contains both "
                "'dataset/' and 'factory-setting/'."
            )
        p = p.parent


RIFTS_ROOT: Path = _find_rifts_root(Path(__file__).parent)
STATE_ROOT: Path = RIFTS_ROOT / "current-state"
FACTORY_ROOT: Path = RIFTS_ROOT / "factory-setting"
DATA_ROOT: Path = RIFTS_ROOT / "dataset" / "Univariate_ts"
SRC_ROOT: Path = RIFTS_ROOT / "src"
SCRIPTS_ROOT: Path = RIFTS_ROOT / "scripts"
RFXPL_ROOT: Path = RIFTS_ROOT / "third_party" / "RFxpl"

for _p in (SRC_ROOT, SCRIPTS_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
