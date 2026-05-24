"""Python 3.10+ compatibility shim for RFxpl's dependencies.

RFxpl pulls in `namedlist`, which still references `collections.Mapping`,
`collections.Sequence`, etc. — names that moved to `collections.abc` in
Python 3.10. Importing this module re-exposes them on `collections` so the
RFxpl import chain succeeds.

Use:
    from max_iaxp import compat  # noqa: F401, must precede any RFxpl import
"""
from __future__ import annotations

import collections
import collections.abc

for _name in dir(collections.abc):
    if not _name.startswith("_") and not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))
