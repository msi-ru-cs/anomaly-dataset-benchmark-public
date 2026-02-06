#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_train_and_report.sh [--mode train|eval|auto] [--config PATH] [--tag TAG]
                          [--run_dir PATH] [--detector err|lik|md] [--profiles LIST]

Positional compat (for your old calls):
  run_train_and_report.sh config/config.yaml mybest

Flags:
  -m, --mode       train | eval | auto   (default: auto)
                   auto = train if --run_dir not given, else eval-only.
  -c, --config     Path to YAML config (default: config/config.yaml)
  -t, --tag        Optional tag appended to run_id during training
  -r, --run_dir    Evaluate this run_dir (skips training)
  -d, --detector   Override detector (err|lik|md). Default = config.report.detector
  -p, --profiles   Comma list of profile names to evaluate (e.g., standard,low_fn,low_fp).
                   Default = profiles in config.report.profiles
  -h, --help       Show this help

Notes:
  * Training now goes through `python -m src.main` so placeholders like
    ${dataset.name} are resolved before the trainer runs.

Outputs:
  - Writes/updates <run_dir>/overall_events__*.csv, events_aggregate_profiles.json
  - Appends one row to <run_dir>/nab_table.csv with Window/Layers/Cells/Epocs/BatchSize and the NAB scores.
USAGE
  exit 1
}

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


# Defaults
MODE="auto"
CONFIG="config/config.yaml"
TAG=""
RUN_DIR=""
DETECTOR=""
PROFILES=""

# Parse args (supports old positional: CONFIG TAG)
SEEN_POS1=""
SEEN_POS2=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--mode) MODE="${2:?}"; shift 2;;
    -c|--config) CONFIG="${2:?}"; shift 2;;
    -t|--tag) TAG="${2:-}"; shift 2;;
    -r|--run_dir) RUN_DIR="${2:?}"; shift 2;;
    -d|--detector) DETECTOR="${2:?}"; shift 2;;
    -p|--profiles) PROFILES="${2:?}"; shift 2;;
    -h|--help) usage;;
    *)
      # positional compat: config then tag
      if [[ -z "$SEEN_POS1" ]]; then CONFIG="$1"; SEEN_POS1=1; shift
      elif [[ -z "$SEEN_POS2" ]]; then TAG="$1"; SEEN_POS2=1; shift
      else
        echo "Unknown arg: $1"; usage
      fi
      ;;
  esac
done

# Pull detector from config if not overridden
if [[ -z "$DETECTOR" ]]; then
  DETECTOR="$(python - <<'PY' "$CONFIG"
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1],encoding='utf-8'))
print(cfg.get('report',{}).get('detector','err'))
PY
)"
fi

# Decide mode
if [[ "$MODE" == "auto" ]]; then
  if [[ -n "$RUN_DIR" ]]; then MODE="eval"; else MODE="train"; fi
fi

if [[ "$MODE" == "train" ]]; then
  echo "[1/3] Training via src.main with ${CONFIG} ${TAG:+(tag=$TAG)}"
  # IMPORTANT: go through src.main so ${dataset.name} gets resolved for trainer
  python -m src.main --config "$CONFIG" ${TAG:+--tag "$TAG"}

  echo "[2/3] Locating latest run_dir"
  RUN_DIR="$(ls -dt runs/*/* | head -1 || true)"
  if [[ -z "$RUN_DIR" ]]; then
    echo "[FATAL] No runs found under runs/*/*" >&2; exit 2
  fi
  echo "Using run_dir: $RUN_DIR"

elif [[ "$MODE" == "eval" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    echo "[eval-only] No --run_dir given; picking latest under runs/*/*"
    RUN_DIR="$(ls -dt runs/*/* | head -1 || true)"
    if [[ -z "$RUN_DIR" ]]; then
      echo "[FATAL] No runs found under runs/*/*" >&2; exit 2
    fi
  fi
  echo "[Eval-only] Using run_dir: $RUN_DIR"
else
  echo "[FATAL] --mode must be train|eval|auto, got: $MODE" >&2; exit 2
fi

echo "[3/3] Evaluating NAB-Score metrics (detector=$DETECTOR) and writing table row"
EVAL_ARGS=( --run_dir "$RUN_DIR" --config "$CONFIG" --detector "$DETECTOR" --test_only --write_table )
# If profiles override supplied, pass it; otherwise eval_events will use config.report.profiles
if [[ -n "$PROFILES" ]]; then EVAL_ARGS+=( --profiles "$PROFILES" ); fi

python -m src.pipelines.eval_events "${EVAL_ARGS[@]}"

echo
echo "[DONE]"
echo "  Run dir:         $RUN_DIR"
echo "  NAB score table: $RUN_DIR/nab_score_table.csv"
echo "  Per-profile CSV: $RUN_DIR/overall_events__*.csv"
echo "  Aggregate JSON:  $RUN_DIR/events_aggregate_profiles.json"
