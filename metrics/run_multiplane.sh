#!/usr/bin/env bash
# Multiplane inference, then photometric metrics.
#
#   bash metrics/run_multiplane.sh nuscenes single
#   bash metrics/run_multiplane.sh ddad multiframes --skip-infer
#
# Each saved frame is three pinhole planes from the main camera, yawed by one
# horizontal field of view and stitched left-center-right:
#   outputs/<dataset>_multiplane/<scene>/rgb/{frame}_5_wide.jpg
#   outputs/<dataset>_multiplane_multiframes/<scene>/rgb/{newest}_5_wide.jpg
# Do not pass --output-dir. Metrics read those directories.
#
# Ground truth is sparseMultiplaneImages3 (three panels), not the single
# 3x-wide pinhole GT. WideDrive has no such GT here, so its photometric step
# is skipped unless --gt-root is set.
# CBSR and PD are not run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash metrics/run_multiplane.sh DATASET MODE [options] [inference flags...]

DATASET:
  nuscenes | ddad | lyft1920 | lyft1224 | widedrive

MODE:
  single         one frame per forward, outputs/<dataset>_multiplane
  multiframes    3-frame history window, outputs/<dataset>_multiplane_multiframes

Options:
  --load-from PATH     default: pretrained/model.safetensors
  --skip-infer         score existing renders only
  --gt-root PATH       override the sparseMultiplaneImages3 directory
  --val-list PATH      override the scene list

Other flags are forwarded to inference. Do not pass --output-dir.
EOF
}

default_gt() {
  case "$1" in
    nuscenes) printf '%s' "data/nuscenes/sparseMultiplaneImages3_1554x294" ;;
    ddad) printf '%s' "data/ddad/sparseMultiplaneImages3_1554x322" ;;
    lyft1920) printf '%s' "data/lyft/1920_sparseMultiplaneWideFOVImages3" ;;
    lyft1224) printf '%s' "data/lyft/1224_sparseMultiplaneWideFOVImages3" ;;
    widedrive) printf '%s' "" ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

take_value() {
  local flag="$1"
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == -* ]]; then
    echo "missing value for ${flag}" >&2
    exit 1
  fi
  printf '%s' "$2"
}

if [[ $# -lt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="$1"
shift
case "$DATASET" in
  nuscenes|ddad|lyft1920|lyft1224|widedrive) ;;
  *)
    echo "unknown dataset: $DATASET" >&2
    usage >&2
    exit 1
    ;;
esac

MODE="$1"
shift
case "$MODE" in
  single|multiframes) ;;
  *)
    echo "mode must be single or multiframes, got: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac

LOAD_FROM="pretrained/model.safetensors"
SKIP_INFER=0
GT_ROOT=""
GT_SET=0
VAL_LIST=""
INFER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --load-from)
      LOAD_FROM="$(take_value "$1" "${2:-}")"
      shift 2
      ;;
    --load-from=*)
      LOAD_FROM="${1#*=}"
      shift
      ;;
    --skip-infer)
      SKIP_INFER=1
      shift
      ;;
    --gt-root)
      GT_ROOT="$(take_value "$1" "${2:-}")"
      GT_SET=1
      shift 2
      ;;
    --gt-root=*)
      GT_ROOT="${1#*=}"
      GT_SET=1
      shift
      ;;
    --val-list)
      VAL_LIST="$(take_value "$1" "${2:-}")"
      shift 2
      ;;
    --val-list=*)
      VAL_LIST="${1#*=}"
      shift
      ;;
    --output-dir|--output-dir=*)
      echo "Do not pass --output-dir. Metrics read outputs/${DATASET}_multiplane[_multiframes]." >&2
      exit 1
      ;;
    *)
      INFER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$GT_SET" -eq 0 ]]; then
  GT_ROOT="$(default_gt "$DATASET")"
fi
if [[ "$SKIP_INFER" -eq 0 && ! -e "$LOAD_FROM" ]]; then
  echo "pretrained path not found: $LOAD_FROM" >&2
  exit 1
fi

if [[ "$MODE" == "single" ]]; then
  INFER_SCRIPT="scripts/inference_${DATASET}_multiplane.py"
  RENDER_ROOT="outputs/${DATASET}_multiplane"
else
  INFER_SCRIPT="scripts/inference_${DATASET}_multiplane_multiframes.py"
  RENDER_ROOT="outputs/${DATASET}_multiplane_multiframes"
fi

cleanup_match_dirs() {
  local root="$1"
  local scene_dir name target
  [[ -d "$root" ]] || return 0
  for scene_dir in "$root"/*/; do
    [[ -d "$scene_dir" ]] || continue
    for name in matched_img match; do
      target="${scene_dir}${name}"
      if [[ -d "$target" ]]; then
        rm -rf "$target"
        echo "Removed temp $target"
      fi
    done
  done
}

cleanup_on_exit() {
  if [[ -n "${RENDER_ROOT:-}" ]]; then
    cleanup_match_dirs "$RENDER_ROOT"
  fi
}
trap cleanup_on_exit EXIT

echo "Mode: $MODE -> $RENDER_ROOT"
if [[ "$SKIP_INFER" -eq 0 ]]; then
  if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "inference script not found: $INFER_SCRIPT" >&2
    exit 1
  fi
  echo "Inference -> $INFER_SCRIPT"
  if ((${#INFER_ARGS[@]})); then
    python "$INFER_SCRIPT" --dataset "$DATASET" --load-from "$LOAD_FROM" "${INFER_ARGS[@]}"
  else
    python "$INFER_SCRIPT" --dataset "$DATASET" --load-from "$LOAD_FROM"
  fi
fi

if [[ -z "$GT_ROOT" ]]; then
  echo "No sparseMultiplaneImages3 GT for $DATASET; skip photometric. Pass --gt-root to score." >&2
  exit 0
fi
if [[ ! -d "$GT_ROOT" && "$GT_ROOT" == *"/ddad/"* ]]; then
  alt="${GT_ROOT/\/ddad\//\/ddad_process\/}"
  if [[ -d "$alt" ]]; then
    echo "GT root relocated: $GT_ROOT -> $alt"
    GT_ROOT="$alt"
  fi
fi
if [[ ! -d "$GT_ROOT" ]]; then
  echo "GT root missing, skip photometric: $GT_ROOT" >&2
  exit 0
fi

val_args=()
if [[ -n "$VAL_LIST" ]]; then
  val_args=(--val-list "$VAL_LIST")
fi
echo "Photometric -> $RENDER_ROOT against $GT_ROOT"
python metrics/eval_photometric.py \
  --dataset "$DATASET" \
  --render-root "$RENDER_ROOT" \
  --gt-root "$GT_ROOT" \
  "${val_args[@]}"
echo "Histogram match -> $RENDER_ROOT"
python metrics/eval_photometric.py \
  --dataset "$DATASET" \
  --render-root "$RENDER_ROOT" \
  --gt-root "$GT_ROOT" \
  --histogram-match \
  "${val_args[@]}"

if [[ "$DATASET" == "lyft1920" || "$DATASET" == "lyft1224" ]]; then
  echo "Lyft combined summary (frame-count weighted) -> outputs/lyft1920_multiplane*"
  python metrics/merge_lyft.py --mode "$MODE" --layout multiplane
fi
