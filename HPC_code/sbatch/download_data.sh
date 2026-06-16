#!/usr/bin/env bash
#
# download_data.sh — fetch the PISCOt .nc source products and extract them to the
# per-point monthly parquet tree the pipeline consumes
# ({BASE}/{var}/Outputs/{var}_daily_YYYY_MM.parquet).
#
# Honest, version-explicit folder names (the legacy local folders were labeled OPPOSITE to
# their real PISCOt version: old `tmin` was actually v1.2 and old `tmin_v1` was actually v1.1,
# both confirmed bit-for-bit against figshare). Each article stores daily data as ONE FILE
# PER YEAR (not a single zip):
#   tmin_v11  PISCOt v1.1 TMIN  article 16372509 v1   tmin_daily_YYYY.zip  1981-2016 (~46 GB)
#   td        PISCOt v1.1 TDEW  article 16305341 v1   td_daily_YYYY.zip    1981-2016 (~50 GB)
#   tmin_v12  PISCOt v1.2 TMIN  article 20533715 v2   tmin_daily_YYYY.nc   1981-2020 (~56 GB)
# The v11-vs-v12 TMIN comparison overlaps 1981-2016 (v1.1 + TDEW end in 2016).
#
# We use the figshare API to list each article's files and download them DIRECTLY
# (resumable, no async-zip "202 Accepted" wait). *_mean_*.nc and README files are skipped.
# v1.1 ships yearly .zip (unzipped in place); v1.2 ships bare yearly .nc. The extractor
# (nc_to_point_parquet.py) then samples every daily layer at the CENAGRO potato centroids
# (default) or keeps the full PISCO grid (PERU_POTATO=0).
#
# Idempotent: wget -c resumes; the extractor skips months whose parquet exists (OVERWRITE=1
# to force). Use YEAR_RANGE to fetch a subset (cheap end-to-end test on one year).
#
# Usage:
#   bash HPC_code/sbatch/download_data.sh                       # all 3 vars, all years
#   VARS="td" YEAR_RANGE="1981,1981" bash .../download_data.sh  # one var, one year (test)
#   PERU_POTATO=0 bash .../download_data.sh                     # full ~2M-point grid
#   PURGE_RAW=1   bash .../download_data.sh                     # delete zip/nc after extract
#   BASE=/scratch/$USER/pisco bash .../download_data.sh         # KHIPU scratch
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- config (env-overridable) ---
BASE="${BASE:-/media/ppalacios/Data/henry_simcast_peru}"
RAW="${RAW:-${BASE}/_raw}"
SHP="${SHP:-${BASE}/PotatoZonning/CENAGRO_OnlyPotatoes_Pisco_Altitude.shp}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
VARS="${VARS:-tmin_v11 td tmin_v12}"
PERU_POTATO="${PERU_POTATO:-1}"     # 1 = potato points (default); 0 = full grid
OVERWRITE="${OVERWRITE:-0}"         # 1 = re-extract existing months
PURGE_RAW="${PURGE_RAW:-0}"         # 1 = remove each raw file after it is extracted
YEAR_RANGE="${YEAR_RANGE:-}"        # e.g. 1981,2020 ; empty = all years

# --- TODO (KHIPU): environment activation if not using $PYTHON directly ---
# module load python/3.13 ; source /path/to/.venv/bin/activate

# var -> figshare API URL (pinned to the right article version)
api_url_for() {
  case "$1" in
    tmin_v11) echo "https://api.figshare.com/v2/articles/16372509" ;;            # PISCOt v1.1 TMIN
    td)       echo "https://api.figshare.com/v2/articles/16305341" ;;            # PISCOt v1.1 TDEW
    tmin_v12) echo "https://api.figshare.com/v2/articles/20533715/versions/2" ;; # PISCOt v1.2 TMIN
    *)        echo "" ;;
  esac
}

potato_flag() { [ "${PERU_POTATO}" = "1" ] && echo "--peru-potato" || echo "--no-peru-potato"; }

# List "<download_url>\t<name>" for the daily data files of an article (mean/README skipped,
# optionally filtered to YEAR_RANGE by the 4-digit year in the filename).
# (curl writes the JSON to a temp file; the heredoc occupies stdin as the python program.)
list_files() {
  local tmpf; tmpf="$(mktemp)"
  curl -fsSL "$1" -o "${tmpf}"
  "${PYTHON}" - "${tmpf}" "${YEAR_RANGE}" <<'PY'
import sys, json, re
path = sys.argv[1]
yr = sys.argv[2] if len(sys.argv) > 2 else ""
lo = hi = None
if yr:
    a, b = yr.split(","); lo, hi = int(a), int(b)
with open(path) as fh:
    data = json.load(fh)
for f in data.get("files", []):
    name = f["name"]
    low = name.lower()
    if "daily" not in low or not low.endswith((".nc", ".zip")) or "mean" in low:
        continue
    m = re.search(r"(\d{4})", name)
    if lo is not None and m and not (lo <= int(m.group(1)) <= hi):
        continue
    print(f["download_url"] + "\t" + name)
PY
  rm -f "${tmpf}"
}

mkdir -p "${RAW}"
echo "BASE=${BASE}"
echo "RAW=${RAW}"
echo "potato=$( [ "${PERU_POTATO}" = "1" ] && echo points || echo full-grid )  vars='${VARS}'  year_range='${YEAR_RANGE:-all}'"
df -h "${BASE}" || true

for var in ${VARS}; do
  api="$(api_url_for "${var}")"
  if [ -z "${api}" ]; then echo "!! unknown var '${var}', skipping"; continue; fi
  ncdir="${RAW}/${var}"
  mkdir -p "${ncdir}"
  echo "=== ${var}  (${api}) ==="

  mapfile -t entries < <(list_files "${api}")
  if [ "${#entries[@]}" -eq 0 ]; then
    echo "!! no daily files listed for ${var} (year_range='${YEAR_RANGE:-all}')"; continue
  fi
  echo "[${var}] ${#entries[@]} yearly file(s) to fetch"

  # 1) download each yearly file directly (resume-safe); unzip the v1.1 yearly zips.
  for e in "${entries[@]}"; do
    url="${e%%$'\t'*}"; name="${e##*$'\t'}"
    dest="${ncdir}/${name}"
    echo "[${var}] downloading ${name}"
    wget -c -O "${dest}" "${url}"
    case "${name}" in
      *.zip)
        unzip -o "${dest}" -d "${ncdir}" >/dev/null
        [ "${PURGE_RAW}" = "1" ] && rm -f "${dest}"
        ;;
    esac
  done

  # 2) extract the whole var dir to point parquet (one pass over all years).
  extra=()
  [ "${OVERWRITE}" = "1" ] && extra+=(--overwrite)
  [ -n "${YEAR_RANGE}" ] && extra+=(--year-range "${YEAR_RANGE}")
  echo "[${var}] extracting -> ${BASE}/${var}/Outputs"
  "${PYTHON}" "${REPO_ROOT}/HPC_code/nc_to_point_parquet.py" \
      --var "${var}" --nc-dir "${ncdir}" --base "${BASE}" --shp "${SHP}" \
      "$(potato_flag)" "${extra[@]}"

  # 3) reclaim disk if asked.
  if [ "${PURGE_RAW}" = "1" ]; then
    echo "[${var}] purging raw .nc under ${ncdir}"
    rm -rf "${ncdir}"
  fi
done

echo "All requested variables done."
