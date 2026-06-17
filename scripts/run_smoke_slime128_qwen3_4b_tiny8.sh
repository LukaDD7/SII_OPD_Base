#!/usr/bin/env bash
set -eo pipefail

export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export SLIME_REPO="$LZY_ROOT/repos/slime"
export MEGATRON_REPO="$LZY_ROOT/repos/Megatron-LM"

export HF_MODEL="$LZY_ROOT/models/Qwen3-4B-Instruct-2507"
export TORCH_DIST="$LZY_ROOT/models/Qwen3-4B-Instruct-2507_torch_dist"
export TRAIN_JSONL="$LZY_ROOT/dataset/standard_rl/train_tiny8_slime.jsonl"
export OUT_DIR="$LZY_ROOT/checkpoints/slime128_qwen3_4b_tiny8_smoke"

cd "$LZY_ROOT"
source activate_lzy.sh
slime128

cd "$SLIME_REPO"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NUM_GPUS="${NUM_GPUS:-2}"
export TP_SIZE="${TP_SIZE:-2}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true

export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_USE_DEEP_GEMM=0
export DISABLE_DEEP_GEMM=1
export CUDA_MODULE_LOADING=LAZY

export NO_PROXY="127.0.0.1,localhost,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${USER:-root}-slime128}"
mkdir -p "$TRITON_CACHE_DIR"

export CUDA_HOME="$CONDA_PREFIX/cuda-home"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LZY_ROOT/shared_libs:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

echo "[smoke] HF_MODEL=$HF_MODEL"
echo "[smoke] TORCH_DIST=$TORCH_DIST"
echo "[smoke] TRAIN_JSONL=$TRAIN_JSONL"
echo "[smoke] OUT_DIR=$OUT_DIR"
echo "[smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[smoke] NUM_GPUS=$NUM_GPUS"
echo "[smoke] TP_SIZE=$TP_SIZE"

test -d "$HF_MODEL"
test -f "$TORCH_DIST/latest_checkpointed_iteration.txt"
test -f "$TRAIN_JSONL"

python - <<'PY'
import os, torch
print("torch", torch.__version__, torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(i, torch.cuda.get_device_name(i), "free_gb", round(free/1024**3, 2), "total_gb", round(total/1024**3, 2))
PY

mkdir -p "$OUT_DIR"

pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang.srt" 2>/dev/null || true
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f "raylet|gcs_server|dashboard|runtime_env_agent" 2>/dev/null || true
sleep 3

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT:-0}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

source "$SLIME_REPO/scripts/models/qwen3-4B.sh"

CKPT_ARGS=(
  --hf-checkpoint "$HF_MODEL"
  --ref-load "$TORCH_DIST"
  --load "$OUT_DIR"
  --save "$OUT_DIR"
  --save-interval 1
)

ROLLOUT_ARGS=(
  --prompt-data "$TRAIN_JSONL"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rm-type deepscaler
  --num-rollout 1
  --rollout-batch-size 2
  --n-samples-per-prompt 2
  --num-steps-per-rollout 1
  --rollout-max-response-len 64
  --rollout-temperature 0.7
  --global-batch-size 4
  --balance-data
)

PERF_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --transformer-impl local
  --no-rope-fusion
  --no-persist-layer-norm
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.25
  --sglang-disable-cuda-graph
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start \
  --head \
  --node-ip-address "$MASTER_ADDR" \
  --num-gpus "$NUM_GPUS" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON=$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "$MEGATRON_REPO:$SLIME_REPO:${PYTHONPATH:-}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "$HAS_NVLINK",
    "CUDA_HOME": "$CUDA_HOME",
    "CUDA_PATH": "$CUDA_PATH",
    "PATH": "$PATH",
    "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH",
    "LIBRARY_PATH": "${LIBRARY_PATH:-}",
    "CPATH": "${CPATH:-}",
    "C_INCLUDE_PATH": "${C_INCLUDE_PATH:-}",
    "CPLUS_INCLUDE_PATH": "${CPLUS_INCLUDE_PATH:-}",
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_USE_DEEP_GEMM": "0",
    "DISABLE_DEEP_GEMM": "1",
    "CUDA_MODULE_LOADING": "LAZY",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "NO_PROXY": "$NO_PROXY",
    "no_proxy": "$no_proxy",
    "TRITON_CACHE_DIR": "$TRITON_CACHE_DIR"
  }
}
JSON
)

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$NUM_GPUS" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"

echo "[smoke] ray job finished"
