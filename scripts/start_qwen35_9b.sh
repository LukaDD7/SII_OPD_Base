#!/bin/bash
# ==============================================================================
# start_qwen35_9b.sh — Qwen3.5-9B as sglang server (rollout)
# ==============================================================================
# Qwen3.5-9B 是 multimodal(架构 Qwen3_5ForConditionalGeneration),含 vision
# 替换用户之前提的 Qwen3-VL-8B-Instruct
#
# 用法:
#   bash start_qwen35_9b.sh 30002
# ==============================================================================

set -e
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy

source $LZY_ROOT/miniconda3/etc/profile.d/conda.sh
conda activate agentic-opd-train

export HF_HOME=$LZY_ROOT/.cache/huggingface
export TRITON_CACHE_DIR=$LZY_ROOT/.cache/triton
export TORCH_HOME=$LZY_ROOT/.cache/torch
mkdir -p $TRITON_CACHE_DIR $TORCH_HOME

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LZY_ROOT/shared_libs:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

python -c "import ctypes; ctypes.CDLL('libnuma.so.1'); print('[OK] libnuma loaded')" || {
    echo "ERROR: libnuma.so.1 not found!"; exit 1
}

PORT=${1:-30002}
LOG_DIR=$LZY_ROOT/logs/agentic-opd
mkdir -p $LOG_DIR
LOG=$LOG_DIR/sglang_qwen35_9b_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] Launching Qwen3.5-9B, port=$PORT, log=$LOG"

MODEL_PATH=$LZY_ROOT/models/Qwen3.5-9B

# Qwen3.5-9B 19GB,1 卡就够(single GPU rollout,其他 7 卡可留给 trainer)
export CUDA_VISIBLE_DEVICES=0
export SGLANG_TP_SIZE=1
export SGLANG_KV_CACHE_MEM_FRACTION=0.85
export SGLANG_USE_DEEP_GEMM=1
export SGLANG_ENABLE_FLASHINFER_GEMM=1

python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --context-length 16384 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    2>&1 | tee $LOG
