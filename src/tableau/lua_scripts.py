"""The five Lua scripts that mediate every multi-key tableau transition.

Per ``code/docs/tableau_outline.md`` §18.5 these are the only writes the
worker performs that touch more than one Redis key; each one is one
``EVAL`` round trip so a crashing worker can never leave the search in a
half-updated state.

Conventions:

  * ``KEYS`` always carries the structural keys (leaves, inflight, internals,
    best, node prefix, claim prefix, …) in the order documented at the top
    of each script.
  * ``ARGV`` carries the per-call payload: node_id, child blobs, ρ values,
    timestamps, etc. Child blobs are JSON-encoded ``{id, icf, icf_lo,
    icf_hi, class, rho_cur, rho_max, rho_min, status}`` dicts.

The Python-side wrapper at the bottom (``LuaScripts``) registers each
script via ``r.register_script`` so the actual call site is a normal
Python method.
"""
from __future__ import annotations

from dataclasses import dataclass

import redis


# ---------------------------------------------------------------------------
# 1. claim_top_leaf
# ---------------------------------------------------------------------------
LUA_CLAIM_TOP_LEAF = r"""
-- KEYS[1] = tableau:leaves         (ZSET, score = -rho_max)
-- KEYS[2] = tableau:inflight       (ZSET, score = claim_ts)
-- KEYS[3] = tableau:claim_prefix   (string prefix, append node_id)
-- ARGV[1] = worker_id
-- ARGV[2] = ttl_seconds
-- ARGV[3] = now (epoch seconds)
--
-- Returns:  {node_id, score}  on success
--           nil               if no claimable leaf (rare race or empty)
local top = redis.call('ZPOPMIN', KEYS[1])
if #top == 0 then return nil end
local node_id = top[1]
local score   = top[2]
local claim_key = KEYS[3] .. node_id
if redis.call('SET', claim_key, ARGV[1], 'NX', 'EX', ARGV[2]) == false then
    -- claim conflict: requeue at original score
    redis.call('ZADD', KEYS[1], score, node_id)
    return nil
end
redis.call('ZADD', KEYS[2], ARGV[3], node_id)
return {node_id, score}
"""


# ---------------------------------------------------------------------------
# 2. commit_good
# ---------------------------------------------------------------------------
LUA_COMMIT_GOOD = r"""
-- KEYS[1] = tableau:leaves
-- KEYS[2] = tableau:inflight
-- KEYS[3] = tableau:internals
-- KEYS[4] = tableau:best
-- KEYS[5] = node_prefix     (then ARGV[2] / ARGV[3] = parent/child ids)
-- KEYS[6] = claim_prefix
-- ARGV[1]  = parent_node_id
-- ARGV[2]  = child_node_id
-- ARGV[3]  = child JSON payload (icf, icf_lo, icf_hi, class)
-- ARGV[4]  = child rho_max  (used for -score in tableau:leaves)
-- ARGV[5]  = child rho_cur
-- ARGV[6]  = child rho_min
-- ARGV[7]  = parent_new_rho_cur OR ""  (empty string means "no incumbent update")
-- ARGV[8]  = parent_icf_canon   (only used on incumbent update)
--
-- 1. parent → tableau:internals; status = "internal (advanced)"
-- 2. child  → tableau:leaves at score -rho_max
-- 3. child  → tableau:node:<child_id>  (HMSET payload + rho values + status="leaf")
-- 4. remove parent from tableau:inflight; DEL claim
-- 5. CAS update tableau:best iff parent_new_rho_cur > best.rho

local parent_id = ARGV[1]
local child_id  = ARGV[2]

-- parent → internals
redis.call('SADD', KEYS[3], parent_id)
local parent_key = KEYS[5] .. parent_id
redis.call('HSET', parent_key, 'status', 'internal_advanced')

-- child node payload
local child_key = KEYS[5] .. child_id
local payload = cjson.decode(ARGV[3])
local args = {child_key}
for k, v in pairs(payload) do
    table.insert(args, k); table.insert(args, tostring(v))
end
table.insert(args, 'rho_max'); table.insert(args, ARGV[4])
table.insert(args, 'rho_cur'); table.insert(args, ARGV[5])
table.insert(args, 'rho_min'); table.insert(args, ARGV[6])
table.insert(args, 'status');  table.insert(args, 'leaf')
redis.call('HSET', unpack(args))

-- child → leaves at -rho_max
redis.call('ZADD', KEYS[1], -tonumber(ARGV[4]), child_id)

-- parent → out of inflight + claim DEL
redis.call('ZREM', KEYS[2], parent_id)
redis.call('DEL', KEYS[6] .. parent_id)

-- CAS best on incumbent update
if ARGV[7] ~= '' then
    local new_rho = tonumber(ARGV[7])
    local cur = redis.call('HGET', KEYS[4], 'rho')
    if (cur == false) or (new_rho > tonumber(cur)) then
        redis.call('HSET', KEYS[4],
                   'rho', tostring(new_rho),
                   'id',  parent_id,
                   'icf', ARGV[8])
    end
end
return 1
"""


# ---------------------------------------------------------------------------
# 3. commit_bad
# ---------------------------------------------------------------------------
LUA_COMMIT_BAD = r"""
-- KEYS[1] = tableau:leaves
-- KEYS[2] = tableau:inflight
-- KEYS[3] = tableau:internals
-- KEYS[4] = node_prefix
-- KEYS[5] = claim_prefix
-- ARGV[1]  = parent_node_id
-- ARGV[2]  = child_A_id
-- ARGV[3]  = child_A payload JSON
-- ARGV[4]  = child_A rho_max
-- ARGV[5]  = child_A rho_cur
-- ARGV[6]  = child_A rho_min
-- ARGV[7]  = child_B_id
-- ARGV[8]  = child_B payload JSON
-- ARGV[9]  = child_B rho_max
-- ARGV[10] = child_B rho_cur
-- ARGV[11] = child_B rho_min
--
-- 1. parent → internals, status = "internal_split"
-- 2. two children inserted at -rho_max
-- 3. parent removed from inflight; claim DEL

local parent_id = ARGV[1]
redis.call('SADD', KEYS[3], parent_id)
redis.call('HSET', KEYS[4] .. parent_id, 'status', 'internal_split')

local function write_child(cid, payload_json, rho_max, rho_cur, rho_min)
    local payload = cjson.decode(payload_json)
    local args = {KEYS[4] .. cid}
    for k, v in pairs(payload) do
        table.insert(args, k); table.insert(args, tostring(v))
    end
    table.insert(args, 'rho_max'); table.insert(args, rho_max)
    table.insert(args, 'rho_cur'); table.insert(args, rho_cur)
    table.insert(args, 'rho_min'); table.insert(args, rho_min)
    table.insert(args, 'status');  table.insert(args, 'leaf')
    redis.call('HSET', unpack(args))
    redis.call('ZADD', KEYS[1], -tonumber(rho_max), cid)
end

write_child(ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6])
-- only insert child_B if it is genuinely distinct from child_A (the
-- single-side fall-back in cicf.bipartite_shrinkage emits the same node)
if ARGV[7] ~= ARGV[2] then
    write_child(ARGV[7], ARGV[8], ARGV[9], ARGV[10], ARGV[11])
end

redis.call('ZREM', KEYS[2], parent_id)
redis.call('DEL', KEYS[5] .. parent_id)
return 1
"""


# ---------------------------------------------------------------------------
# 4. commit_closure
# ---------------------------------------------------------------------------
LUA_COMMIT_CLOSURE = r"""
-- KEYS[1] = tableau:inflight
-- KEYS[2] = tableau:internals
-- KEYS[3] = tableau:best
-- KEYS[4] = node_prefix
-- KEYS[5] = claim_prefix
-- ARGV[1] = node_id
-- ARGV[2] = closure reason  ("saturated" | "bb_pruned")
-- ARGV[3] = new_rho_cur OR ""  (empty = no incumbent update)
-- ARGV[4] = inner_icf_canon (only used on incumbent update)
local node_id = ARGV[1]
redis.call('SADD', KEYS[2], node_id)
redis.call('HSET', KEYS[4] .. node_id,
           'status', 'closed_' .. ARGV[2])
redis.call('ZREM', KEYS[1], node_id)
redis.call('DEL', KEYS[5] .. node_id)
if ARGV[3] ~= '' then
    local new_rho = tonumber(ARGV[3])
    local cur = redis.call('HGET', KEYS[3], 'rho')
    if (cur == false) or (new_rho > tonumber(cur)) then
        redis.call('HSET', KEYS[3],
                   'rho', tostring(new_rho),
                   'id',  node_id,
                   'icf', ARGV[4])
    end
end
return 1
"""


# ---------------------------------------------------------------------------
# 5. gc_inflight
# ---------------------------------------------------------------------------
LUA_GC_INFLIGHT = r"""
-- KEYS[1] = tableau:leaves
-- KEYS[2] = tableau:inflight
-- KEYS[3] = node_prefix
-- KEYS[4] = claim_prefix
-- ARGV[1] = now (epoch seconds)
-- ARGV[2] = ttl (seconds)
--
-- For every (node_id, claim_ts) in tableau:inflight with claim_ts < now-ttl:
--   * DELETE the claim key (it has likely already expired anyway).
--   * Re-add node_id to tableau:leaves at its ORIGINAL -rho_max score
--     (read from the node hash). If the node hash is missing, skip.
--   * Remove the entry from tableau:inflight.
-- Returns the number of nodes reaped.
local cutoff = tonumber(ARGV[1]) - tonumber(ARGV[2])
local stale = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', cutoff)
local n = 0
for _, node_id in ipairs(stale) do
    local rho_max = redis.call('HGET', KEYS[3] .. node_id, 'rho_max')
    if rho_max ~= false then
        redis.call('ZADD', KEYS[1], -tonumber(rho_max), node_id)
    end
    redis.call('DEL', KEYS[4] .. node_id)
    redis.call('ZREM', KEYS[2], node_id)
    n = n + 1
end
return n
"""


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


@dataclass
class LuaScripts:
    """Registered ``Script`` handles for a given Redis client.

    Use ``LuaScripts.bind(r)`` once at worker startup; the wrappers below
    accept the structural keys + payload and return the script result.
    """
    claim_top_leaf: redis.commands.core.Script
    commit_good: redis.commands.core.Script
    commit_bad: redis.commands.core.Script
    commit_closure: redis.commands.core.Script
    gc_inflight: redis.commands.core.Script

    @classmethod
    def bind(cls, r: redis.Redis) -> "LuaScripts":
        return cls(
            claim_top_leaf=r.register_script(LUA_CLAIM_TOP_LEAF),
            commit_good=r.register_script(LUA_COMMIT_GOOD),
            commit_bad=r.register_script(LUA_COMMIT_BAD),
            commit_closure=r.register_script(LUA_COMMIT_CLOSURE),
            gc_inflight=r.register_script(LUA_GC_INFLIGHT),
        )
