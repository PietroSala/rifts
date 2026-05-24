"""Redis connection helper.

DBs follow the legacy DRIFTS schema; for Step 1 only DB 0 (`DATA`) is used.
The other DBs are reserved and will be wired up in later steps.
"""
from __future__ import annotations

import os
from typing import Dict

import redis


DB_INDEX: Dict[str, int] = {
    "DATA":  0,  # indexed forest, EU, trivial ICFs
    "CAN":   1,  # candidate ICFs
    "R":     2,  # confirmed reasons
    "NR":    3,  # confirmed non-reasons
    "CAR":   4,  # candidate anti-reasons
    "AR":    5,  # confirmed anti-reasons
    "GP":    6,  # good profiles (reasons)
    "BP":    7,  # bad profiles (anti-reasons)
    "PR":    8,  # preferred reasons
    "AP":    9,  # anti-reason profiles
}


def get_client(db: str = "DATA",
               host: str | None = None,
               port: int | None = None) -> redis.Redis:
    """Return a `redis.Redis` client on the requested logical DB.

    Host/port default to environment vars `REDIS_HOST` / `REDIS_PORT`, else
    `localhost:6379`. `decode_responses=True` keeps the rest of the codebase
    free of byte-string handling.
    """
    if db not in DB_INDEX:
        raise ValueError(f"unknown db role {db!r}; choose from {list(DB_INDEX)}")
    return redis.Redis(
        host=host or os.environ.get("REDIS_HOST", "localhost"),
        port=int(port or os.environ.get("REDIS_PORT", 6379)),
        db=DB_INDEX[db],
        decode_responses=True,
    )
