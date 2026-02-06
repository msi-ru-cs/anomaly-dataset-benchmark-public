#!/usr/bin/env bash
set -euo pipefail

# --- GPU/CPU auto-select ------------------------------------------------------
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
  GPU_COUNT=$(
    python -c "import sys
try:
    import tensorflow as tf
    print(len(tf.config.list_physical_devices('GPU')))
except Exception:
    print(0)
" 2>/dev/null || echo 0)
  if [ "${GPU_COUNT}" -gt 0 ]; then
    export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
    echo "[GPU] Found ${GPU_COUNT} GPU(s). Using GPU."
  else
    export CUDA_VISIBLE_DEVICES=""
    echo "[GPU] No GPU found. Forcing CPU."
  fi
else
  echo "[GPU] Respecting CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
fi
# -----------------------------------------------------------------------------


# defaults
MODE=""
CONFIG="config/config.yaml"
RUN_DIR=""
DETECTOR=""
TAG=""

# parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)       MODE="${2:-}"; shift 2 ;;
    --config)     CONFIG="${2:-}"; shift 2 ;;
    --run_dir)    RUN_DIR="${2:-}"; shift 2 ;;
    --detector)   DETECTOR="${2:-}"; shift 2 ;;
    --tag)        TAG="${2:-}"; shift 2 ;;
    *)            ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "[ERR] --mode train|eval is required"; exit 2
fi

# build command (still calls run_train_and_report.sh, which now routes training via src.main)
CMD=( "./run_train_and_report.sh" "--mode" "$MODE" "--config" "$CONFIG" )
[[ -n "$RUN_DIR"   ]] && CMD+=( "--run_dir" "$RUN_DIR" )
[[ -n "$DETECTOR"  ]] && CMD+=( "--detector" "$DETECTOR" )
[[ -n "$TAG"       ]] && CMD+=( "--tag" "$TAG" )
# pass any stray args too
CMD+=( "${ARGS[@]}" )

# log + pid files
mkdir -p logs
ts=$(date +'%Y-%m-%d_%H-%M-%S')
cfg_base=$(basename "$CONFIG" .yaml)
det_sfx=${DETECTOR:+_${DETECTOR}}
tag_sfx=${TAG:+_${TAG}}
log="logs/${ts}_${MODE}_${cfg_base}${det_sfx}${tag_sfx}.log"
pidf="logs/${ts}_${MODE}_${cfg_base}${det_sfx}${tag_sfx}.pid"

echo "[BG] starting: ${CMD[*]}"
echo "[BG] log:  $log"
echo "[BG] pid:  $pidf"

# run in background
( nohup "${CMD[@]}" >"$log" 2>&1 & echo $! > "$pidf" ) & disown

echo "[BG] tail the log with:"
echo "     tail -f \"$log\""
echo "[BG] stop with:"
echo "     kill \$(cat \"$pidf\")"
