#!/usr/bin/env bash
set -eo pipefail

export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export SLIME_REPO="$LZY_ROOT/repos/slime"
export MEGATRON_REPO="$LZY_ROOT/repos/Megatron-LM"

cd "$LZY_ROOT"
source activate_lzy.sh
slime128

export HF_MODEL="$LZY_ROOT/models/Qwen3-4B-Instruct-2507"
export TORCH_DIST="$LZY_ROOT/models/Qwen3-4B-Instruct-2507_torch_dist"

export WANDB_MODE=disabled
export WANDB_DISABLED=true

export CUDA_HOME="$CONDA_PREFIX/cuda-home"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LZY_ROOT/shared_libs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

echo "[convert] HF_MODEL=$HF_MODEL"
echo "[convert] TORCH_DIST=$TORCH_DIST"
echo "[convert] SLIME_REPO=$SLIME_REPO"
echo "[convert] MEGATRON_REPO=$MEGATRON_REPO"

test -d "$HF_MODEL"

rm -rf "$TORCH_DIST"

cd "$SLIME_REPO"
source scripts/models/qwen3-4B.sh

PYTHONPATH="$MEGATRON_REPO:$SLIME_REPO:${PYTHONPATH:-}" \
python tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --transformer-impl local \
  --no-rope-fusion \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint "$HF_MODEL" \
  --save "$TORCH_DIST" \
  2>&1 | tee /tmp/slime128_convert_qwen3_4b.out

rc=${PIPESTATUS[0]}
echo "convert exit=$rc"

find "$TORCH_DIST" -maxdepth 4 -type f \
  \( -name '*.distcp' -o -name 'latest_checkpointed_iteration.txt' \) \
  | head -80

exit "$rc"
