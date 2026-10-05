#!/usr/bin/env bash
# UniSplat wide inference, then DWSplat-style metrics, for one dataset.
#
#   conda activate unisplat
#   bash metrics/run_wide.sh ddad single
#   bash metrics/run_wide.sh nuscenes multiframes --skip-infer
#   bash metrics/run_lyft1920.sh single --max-frames 1
#
# Single-frame renders: outputs/<dataset>_wide/<scene>/rgb/{frame}_5_wide.jpg
# Multi-frame renders:  outputs/<dataset>_wide_multiframes/<scene>/rgb/{newest}_5_wide.jpg
# Do not pass --output-dir. Metrics read those two directories.
#
# Photometric and histogram-matched scores need GT.
# CBSR and PD do not. eval_crcs.py and eval_ips.py are kept but not called.
# nuScenes sparse GT is expected at data/nuscenes/sparseMultiplaneImages3_1554x294
# and is not in this checkout; photometric/HM are skipped until that directory exists.
# The first photometric run may download LPIPS AlexNet weights.
# After metrics finish, per-scene matched_img/ and match/ temps are removed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash metrics/run_wide.sh DATASET MODE [options] [inference flags...]

DATASET:
  nuscenes | ddad | lyft1920 | lyft1224 | widedrive

MODE (required; only one runs):
  single         one frame per forward, outputs/<dataset>_wide
  multiframes    3-frame history window, outputs/<dataset>_wide_multiframes

Options:
  --load-from PATH     default: pretrained/model.safetensors
  --skip-infer         score existing renders only
  --gt-root PATH       override the ground-truth directory
  --val-list PATH      override the scene list

Other flags are forwarded to the inference script only
(for example --max-frames 1 or --scene 000). Do not pass --output-dir.

Wrappers:
  bash metrics/run_nuscenes.sh
  bash metrics/run_ddad.sh
  bash metrics/run_lyft1920.sh
  bash metrics/run_lyft1224.sh
  bash metrics/run_widedrive.sh
  bash metrics/run_all.sh
EOF
}

default_gt() {
  case "$1" in
    nuscenes) printf '%s' "data/nuscenes/sparseMultiplaneImages3_1554x294" ;;
    ddad) printf '%s' "data/ddad/sparseMultiplaneImages3_1554x322" ;;
    lyft1920) printf '%s' "data/lyft/1920_sparseMultiplaneWideFOVImages3" ;;
    lyft1224) printf '%s' "data/lyft/1224_sparseMultiplaneWideFOVImages3" ;;
    widedrive) printf '%s' "data/WideDrive_processed/WideDriveVal" ;;
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

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
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

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "missing mode: single or multiframes" >&2
  usage >&2
  exit 1
fi

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
    --single-only|--multi-only)
      echo "pass mode as a positional argument: single or multiframes" >&2
      exit 1
      ;;
    --gt-root)
      GT_ROOT="$(take_value "$1" "${2:-}")"
      shift 2
      ;;
    --gt-root=*)
      GT_ROOT="${1#*=}"
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
      echo "Do not pass --output-dir. Metrics read outputs/${DATASET}_wide and outputs/${DATASET}_wide_multiframes." >&2
      exit 1
      ;;
    *)
      INFER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$GT_ROOT" ]]; then
  GT_ROOT="$(default_gt "$DATASET")"
fi
if [[ "$SKIP_INFER" -eq 0 && ! -e "$LOAD_FROM" ]]; then
  echo "pretrained path not found: $LOAD_FROM" >&2
  exit 1
fi

run_infer() {
  local script="$1"
  if [[ ! -f "$script" ]]; then
    echo "inference script not found: $script" >&2
    exit 1
  fi
  echo "Inference -> $script"
  if ((${#INFER_ARGS[@]})); then
    python "$script" --dataset "$DATASET" --load-from "$LOAD_FROM" "${INFER_ARGS[@]}"
  else
    python "$script" --dataset "$DATASET" --load-from "$LOAD_FROM"
  fi
}

run_metrics() {
  local render_root="$1"
  local val_args=()
  if [[ -n "$VAL_LIST" ]]; then
    val_args=(--val-list "$VAL_LIST")
  fi
  if [[ -d "$GT_ROOT" ]]; then
    echo "Photometric -> $render_root"
    python metrics/eval_photometric.py \
      --dataset "$DATASET" \
      --render-root "$render_root" \
      --gt-root "$GT_ROOT" \
      "${val_args[@]}"
    echo "Histogram match -> $render_root"
    python metrics/eval_photometric.py \
      --dataset "$DATASET" \
      --render-root "$render_root" \
      --gt-root "$GT_ROOT" \
      --histogram-match \
      "${val_args[@]}"
  else
    echo "GT root missing, skip photometric/HM: $GT_ROOT" >&2
  fi
  echo "CBSR and PD -> $render_root"
  python metrics/eval_consistency.py \
    --dataset "$DATASET" \
    --render-root "$render_root" \
    "${val_args[@]}"
}

if [[ "$MODE" == "single" ]]; then
  INFER_SCRIPT="scripts/inference_${DATASET}_wide.py"
  RENDER_ROOT="outputs/${DATASET}_wide"
else
  INFER_SCRIPT="scripts/inference_${DATASET}_wide_multiframes.py"
  RENDER_ROOT="outputs/${DATASET}_wide_multiframes"
fi

cleanup_match_dirs() {
  local root="$1"
  local scene_dir name target removed=0
  [[ -d "$root" ]] || return 0
  for scene_dir in "$root"/*/; do
    [[ -d "$scene_dir" ]] || continue
    for name in matched_img match; do
      target="${scene_dir}${name}"
      if [[ -d "$target" ]]; then
        rm -rf "$target"
        echo "Removed temp $target"
        removed=$((removed + 1))
      fi
    done
  done
  if [[ "$removed" -eq 0 ]]; then
    echo "No per-scene match temp under $root"
  fi
}

echo "Mode: $MODE -> $RENDER_ROOT"
cleanup_on_exit() {
  if [[ -n "${RENDER_ROOT:-}" ]]; then
    cleanup_match_dirs "$RENDER_ROOT"
  fi
}
trap cleanup_on_exit EXIT

if [[ "$SKIP_INFER" -eq 0 ]]; then
  run_infer "$INFER_SCRIPT"
fi
run_metrics "$RENDER_ROOT"
