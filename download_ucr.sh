#!/usr/bin/env bash
# Download the UCR Univariate Time-Series Classification Archive (2018 .ts
# release) into ./dataset/Univariate_ts/ . Idempotent: if the target tree is
# already populated, the script exits silently with status 0.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${HERE}/dataset"
ZIP_PATH="${DATA_DIR}/Univariate2018_ts.zip"
TARGET_DIR="${DATA_DIR}/Univariate_ts"
URL="https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip"

is_populated() {
  # "Populated" means: the target dir exists, contains at least 100 sub-
  # directories (the archive has 128), and at least one of them has a *_TRAIN.ts
  # file. This is a cheap proxy for a successful unpack.
  if [ ! -d "${TARGET_DIR}" ]; then return 1; fi
  local n_subdirs
  n_subdirs=$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
  if [ "${n_subdirs:-0}" -lt 100 ]; then return 1; fi
  if ! find "${TARGET_DIR}" -mindepth 2 -maxdepth 2 -name "*_TRAIN.ts" -print -quit | grep -q .; then
    return 1
  fi
  return 0
}

if is_populated; then
  echo "[ucr] ${TARGET_DIR} already populated; nothing to do."
  exit 0
fi

mkdir -p "${DATA_DIR}"

if [ ! -f "${ZIP_PATH}" ]; then
  echo "[ucr] downloading ${URL}"
  echo "[ucr]   -> ${ZIP_PATH}  (about 125 MB)"
  curl -fL --retry 3 --connect-timeout 20 --progress-bar -o "${ZIP_PATH}" "${URL}"
else
  echo "[ucr] reusing existing archive at ${ZIP_PATH}"
fi

echo "[ucr] unpacking into ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"
# The zip top-level is "Univariate_ts/<dataset>/...". We want the contents
# of "Univariate_ts/" placed directly inside TARGET_DIR. Extract to a temp
# folder and move (so we can rerun safely if the zip top-level changes name).
TMP_DIR="$(mktemp -d "${DATA_DIR}/.unpack.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT
unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

ROOT="$(find "${TMP_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [ -z "${ROOT:-}" ]; then
  echo "[ucr] ERROR: zip did not contain a top-level directory" >&2
  exit 1
fi
# Move every dataset directory from ROOT/ to TARGET_DIR/, overwriting siblings.
find "${ROOT}" -mindepth 1 -maxdepth 1 -exec mv -f {} "${TARGET_DIR}/" \;

if ! is_populated; then
  echo "[ucr] ERROR: unpack finished but ${TARGET_DIR} does not look populated" >&2
  exit 1
fi

echo "[ucr] cleanup: removing ${ZIP_PATH}"
rm -f "${ZIP_PATH}"

n_subdirs=$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "[ucr] done. ${TARGET_DIR} now holds ${n_subdirs} dataset sub-folders."
