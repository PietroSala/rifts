#!/usr/bin/env bash
# Prepare rifts-local/current-state/ before any sweep or the dashboard is run.
#
# Usage:
#   init.sh                       # default: copy from factory-setting/ if current-state/ is absent
#   init.sh --from-factory        # explicit form of the default
#   init.sh --from-scratch        # create an empty current-state/ skeleton
#                                 #   (scripts/run_all.py will then train forests,
#                                 #    fetch baselines, run HPO, build metrics)
#   init.sh --force [--from-...]  # DESTRUCTIVE: wipe an existing current-state/
#                                 #   before re-creating it
#
# After this script returns successfully, every script under rifts-local/scripts/
# reads from and writes to rifts-local/current-state/. The pristine
# rifts-local/factory-setting/ snapshot and the rifts-local/dataset/ archive
# are never touched.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${HERE}/current-state"
FACTORY="${HERE}/factory-setting"
DATASET="${HERE}/dataset/Univariate_ts"

MODE="from-factory"
FORCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-factory) MODE="from-factory"; shift ;;
        --from-scratch) MODE="from-scratch"; shift ;;
        --force)        FORCE="1"; shift ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[init] unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

# Sanity: the UCR archive must be in place. The sweeps and even some helper
# scripts (compute_model_stats.py, the max-iaxp pipeline) read .ts files
# directly, so a missing UCR is a hard error regardless of MODE.
if [ ! -d "${DATASET}" ] || \
   [ -z "$(find "${DATASET}" -mindepth 2 -maxdepth 2 -name '*_TRAIN.ts' -print -quit)" ]; then
    echo "[init] ERROR: ${DATASET} is missing or empty." >&2
    echo "[init]        run ./download_ucr.sh first." >&2
    exit 1
fi

# Refuse to clobber unless --force is passed.
if [ -d "${STATE}" ]; then
    if [ -z "${FORCE}" ]; then
        echo "[init] ${STATE} already exists; nothing to do."
        echo "[init] pass --force to wipe and re-initialise."
        exit 0
    fi
    echo "[init] removing existing ${STATE} (--force)"
    rm -rf "${STATE}"
fi

case "${MODE}" in
    from-factory)
        if [ ! -d "${FACTORY}" ]; then
            echo "[init] ERROR: ${FACTORY} not found." >&2
            exit 1
        fi
        echo "[init] copying factory-setting/ -> current-state/"
        echo "[init]   (~1 GB; rsync output suppressed, this takes about a minute on a local SSD)"
        mkdir -p "${STATE}"
        rsync -a --exclude='__pycache__' --exclude='*.pyc' \
              "${FACTORY}/" "${STATE}/"

        # The factory snapshot ships the sweep DBs in compressed (.db.zst)
        # form to keep them trackable in git. Inflate any .zst whose
        # matching .db is missing.
        shopt -s nullglob
        for zst in "${STATE}"/sweeps/*/sweep.db.zst; do
            db="${zst%.zst}"
            if [ ! -f "${db}" ]; then
                rel="${db#${STATE}/}"
                echo "[init]   inflating ${rel}  (zstd -d)"
                if ! command -v zstd >/dev/null; then
                    echo "[init] ERROR: zstd not on PATH; cannot inflate ${zst}" >&2
                    exit 1
                fi
                zstd -q -d "${zst}" -o "${db}"
            fi
        done
        shopt -u nullglob

        echo "[init] done. current-state/ populated from factory-setting/."
        ;;
    from-scratch)
        echo "[init] creating an empty current-state/ skeleton"
        mkdir -p \
            "${STATE}/models" \
            "${STATE}/metrics" \
            "${STATE}/hpo" \
            "${STATE}/max-iaxp" \
            "${STATE}/sweeps/maximal_reasons" \
            "${STATE}/sweeps/refinements" \
            "${STATE}/experiments_order"
        echo "[init] done. current-state/ is empty."
        echo "[init] next steps:"
        echo "[init]   1. python scripts/run_all.py        # HPO + per-dataset RF training"
        echo "[init]   2. python scripts/compute_model_stats.py"
        echo "[init]   3. python scripts/fetch_reference.py # TSF + 1NN-DTW baselines"
        echo "[init]   4. python scripts/build_hasse.py    # complexity Hasse diagram"
        echo "[init]   5. python scripts/max_iaxp/sweep.py # Max-iAXp baseline sweep"
        echo "[init]   6. python scripts/greedy_cegar_sweep.py"
        echo "[init]   7. python scripts/refinement_doubling_sweeper.py"
        ;;
esac
