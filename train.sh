#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/data/envs/PDF/bin/python"
DATA_ROOT="/data/datasets/PRCC"
OUTPUT_ROOT="/data/outputs/PDF"
CFG="${PDF_CFG:-configs/prcc_2gpu.yaml}"
CAPTION_FILE="${PDF_CAPTION_FILE:-$PROJECT_ROOT/data/captions/prcc.json}"
TAG="${PDF_TAG:-corrected-2gpu}"
MASTER_PORT="${PDF_MASTER_PORT:-12336}"

export CUDA_VISIBLE_DEVICES="0,1"
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "$PROJECT_ROOT"

if [[ "$CFG" = /* ]]; then
    CFG_PATH="$CFG"
else
    CFG_PATH="$PROJECT_ROOT/$CFG"
fi

if [[ "$CAPTION_FILE" = /* ]]; then
    CAPTION_PATH="$CAPTION_FILE"
else
    CAPTION_PATH="$PROJECT_ROOT/$CAPTION_FILE"
fi

test -x "$PYTHON"
test -d "$DATA_ROOT/prcc/rgb/train"
test -d "$DATA_ROOT/prcc/rgb/val"
test -d "$DATA_ROOT/prcc/rgb/test"
test -f "$CAPTION_PATH"
test -f "$CFG_PATH"

case "$CAPTION_PATH" in
    /data/*) ;;
    *) echo "Refusing to use a caption file outside /data: $CAPTION_PATH" >&2; exit 1 ;;
esac

case "$OUTPUT_ROOT" in
    /data/outputs/PDF) ;;
    *) echo "Refusing to use a non-/data output path: $OUTPUT_ROOT" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT_ROOT"

run_status=0
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port "$MASTER_PORT" \
    clip_finetune.py \
    --dataset prcc \
    --cfg "$CFG_PATH" \
    --root "$DATA_ROOT" \
    --caption "$CAPTION_PATH" \
    --output "$OUTPUT_ROOT" \
    --tag "$TAG" || run_status=$?

run_dir=""
if [ -d "$OUTPUT_ROOT/prcc/$TAG" ]; then
    run_dir="$(find "$OUTPUT_ROOT/prcc/$TAG" -mindepth 1 -maxdepth 1 -type d -name 'baseline*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi

if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
    {
        printf 'timestamp_utc: '
        date -u '+%Y-%m-%dT%H:%M:%SZ'
        printf 'project_root: %s\n' "$PROJECT_ROOT"
        printf 'python: %s\n' "$PYTHON"
        printf 'config: %s\n' "$CFG_PATH"
        printf 'tag: %s\n' "$TAG"
        printf 'dataset_root: %s\n' "$DATA_ROOT"
        printf 'caption_file: %s\n' "$CAPTION_PATH"
        printf 'caption_sha256: '
        sha256sum "$CAPTION_PATH" | awk '{print $1}'
        printf 'output_root: %s\n' "$OUTPUT_ROOT"
        printf 'cuda_visible_devices: %s\n' "$CUDA_VISIBLE_DEVICES"
        printf 'nproc_per_node: 2\n'
        printf 'batch_per_gpu: 32\n'
        printf 'global_batch: 64\n'
        printf 'git_commit: '
        git rev-parse HEAD
        printf 'git_status:\n'
        git status --short
        printf '\n--- package versions ---\n'
        "$PYTHON" -c 'import torch, torchvision, timm, yacs; print("torch", torch.__version__); print("torchvision", torchvision.__version__); print("timm", timm.__version__); print("yacs", yacs.__version__ if hasattr(yacs, "__version__") else "installed")'
        printf '\n--- hardware ---\n'
        /usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi || true
    } > "$run_dir/run_manifest.txt"
    git diff --binary > "$run_dir/git_diff.patch" || true
fi

exit "$run_status"
