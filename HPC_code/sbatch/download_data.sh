#!/usr/bin/env bash
#
# download_data.sh — fetch the PISCOt .nc source products and extract them to the
# per-point monthly parquet tree the pipeline consumes
# ({BASE}/{var}/Outputs/{var}_daily_YYYY_MM.parquet).
#
# Source -> variable map (figshare versioned ndownloader URLs):
#   tmin    PISCOt v1.1 TMIN  articles/16372509/versions/1   (~46 GB)
#   td      PISCOt v1.1 TDEW  articles/16305341/versions/1   (~50 GB)
#   tmin_v1 PISCOt v1.2 TMIN  articles/20533715/versions/2   (~56 GB)
#
# Each link is a ZIP of the article's .nc file(s). The extractor (nc_to_point_parquet.py)
# samples every daily layer at the CENAGRO potato centroids (default) or keeps the full
# PISCO grid (PERU_POTATO=0). Idempotent: wget -c resumes partial downloads; the extractor
# skips months whose parquet already exists (unless OVERWRITE=1).
#
# Usage:
#   bash HPC_code/sbatch/download_data.sh                 # all 3 vars, potato points
#   VARS="td"        bash HPC_code/sbatch/download_data.sh    # one var
#   PERU_POTATO=0    bash HPC_code/sbatch/download_data.sh    # full ~2M-point grid
#   PURGE_RAW=1      bash HPC_code/sbatch/download_data.sh    # delete zip+nc after extract
#   BASE=/scratch/$USER/pisco bash HPC_code/sbatch/download_data.sh   # KHIPU scratch
#
set -euo pipefail

# --- repo root (this file is HPC_code/sbatch/download_data.sh) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- config (env-overridable) ---
BASE="${BASE:-/media/ppalacios/Data/henry_simcast_peru}"
RAW="${RAW:-${BASE}/_raw}"
SHP="${SHP:-${BASE}/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.shp}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
VARS="${VARS:-tmin td tmin_v1}"
PERU_POTATO="${PERU_POTATO:-1}"     # 1 = potato points (default); 0 = full grid
OVERWRITE="${OVERWRITE:-0}"         # 1 = re-extract existing months
PURGE_RAW="${PURGE_RAW:-0}"         # 1 = remove zip + extracted .nc after each var
YEAR_RANGE="${YEAR_RANGE:-}"        # e.g. 1981,2020 ; empty = all

# --- TODO (KHIPU): environment activation if not using $PYTHON directly ---
# module load python/3.13 ; source /path/to/.venv/bin/activate

# var -> figshare article/version
url_for() {
  case "$1" in
    tmin)    echo "https://figshare.com/ndownloader/articles/16372509/versions/1" ;;
    td)      echo "https://figshare.com/ndownloader/articles/16305341/versions/1" ;;
    tmin_v1) echo "https://figshare.com/ndownloader/articles/20533715/versions/2" ;;
    *) echo "" ;;
  esac
}

potato_flag() { [ "${PERU_POTATO}" = "1" ] && echo "--peru-potato" || echo "--no-peru-potato"; }

mkdir -p "${RAW}"
echo "BASE=${BASE}"
echo "RAW=${RAW}"
echo "potato=$( [ "${PERU_POTATO}" = "1" ] && echo points || echo full-grid )  vars='${VARS}'"
df -h "${BASE}" || true

for var in ${VARS}; do
  url="$(url_for "${var}")"
  if [ -z "${url}" ]; then echo "!! unknown var '${var}', skipping"; continue; fi

  zip="${RAW}/${var}.zip"
  ncdir="${RAW}/${var}"
  echo "=== ${var} ==="

  # 1) download (resume-safe)
  echo "[${var}] downloading ${url}"
  wget -c -O "${zip}" "${url}"

  # 2) unzip (figshare returns a zip even for a single .nc)
  mkdir -p "${ncdir}"
  if file "${zip}" | grep -qi zip; then
    unzip -o "${zip}" -d "${ncdir}"
  else
    # not a zip (already a bare .nc) — keep it in place
    cp -f "${zip}" "${ncdir}/${var}.nc"
  fi

  # 3) extract to point parquet
  extra=()
  [ "${OVERWRITE}" = "1" ] && extra+=(--overwrite)
  [ -n "${YEAR_RANGE}" ] && extra+=(--year-range "${YEAR_RANGE}")
  echo "[${var}] extracting -> ${BASE}/${var}/Outputs"
  "${PYTHON}" "${REPO_ROOT}/HPC_code/nc_to_point_parquet.py" \
      --var "${var}" --nc-dir "${ncdir}" --base "${BASE}" --shp "${SHP}" \
      "$(potato_flag)" "${extra[@]}"

  # 4) reclaim disk if asked
  if [ "${PURGE_RAW}" = "1" ]; then
    echo "[${var}] purging raw ${zip} and ${ncdir}"
    rm -rf "${zip}" "${ncdir}"
  fi
done

echo "All requested variables done."
