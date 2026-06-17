#!/bin/bash
# ==============================================================================
# start_ds_v4_flash.sh — DeepSeek-V4-Flash as sglang server (teacher)
# ==============================================================================
# 配置源自 projects/deepseek-v4-flash-gym/CLAUDE.md 成功部署:
#   - TP=4, EP=4, KV fp8_e5m2, context 65536
#   - 4 卡 GPU 留给 sglang,另 4 卡留 trainer
#   - libnuma.so.1 在 /lib/x86_64-linux-gnu (NFS 共享)
# 用法:
#   bash start_ds_v4_flash.sh 30000
# 测试:
#   curl http://localhost:30000/v1/models
# ==============================================================================

set -e
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy

# 1. 激活环境
source $LZY_ROOT/miniconda3/etc/profile.d/conda.sh
conda activate agentic-opd-train

# 2. 路径
export HF_HOME=$LZY_ROOT/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$LZY_ROOT/.cache/huggingface/hub
export TRITON_CACHE_DIR=$LZY_ROOT/.cache/triton
export TORCH_HOME=$LZY_ROOT/.cache/torch
export VLLM_CACHE_ROOT=$LZY_ROOT/.cache/vllm
mkdir -p $TRITON_CACHE_DIR $TORCH_HOME

# 3. 编译器(用 env 自带,避免系统 GCC ABI 冲突)
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX

# 4. 共享库(NFS libnuma + env 自带 libstdcxx)
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LZY_ROOT/shared_libs:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# 5. 验证 libnuma
python -c "import ctypes; ctypes.CDLL('libnuma.so.1'); print('[OK] libnuma loaded')" || {
    echo "ERROR: libnuma.so.1 not found!"
    exit 1
}

# 6. 端口
PORT=${1:-30000}
LOG_DIR=$LZY_ROOT/logs/agentic-opd
mkdir -p $LOG_DIR
LOG=$LOG_DIR/sglang_ds_v4_flash_$(date +%Y%m%d_%H%M%S).log
echo "[$(date)] Launching DeepSeek-V4-Flash, port=$PORT, log=$LOG"

# 7. DeepSeek-V4 sglang 启动参数
#    注意:若 sglang 0.5.12.post1 不支持 deepseek_v4 model_type,
#         把 config.json 里的 model_type 临时改 deepseek_v3(slime 官方 kimik2 同款 trick)
MODEL_PATH=$LZY_ROOT/models/deepseek-v4-flash

# 4 卡 TP+EP,2 卡留给 trainer(若你跑纯推理可改 TP=8 EP=8 DP=1)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_TP_SIZE=4
export SGLANG_EP_SIZE=4
export SGLANG_KV_CACHE_MEM_FRACTION=0.82
export SGLANG_USE_DEEP_GEMM=1
export SGLANG_ENABLE_FLASHINFER_GEMM=1

python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size 4 \
    --expert-parallel-size 4 \
    --mem-fraction-static 0.82 \
    --context-length 65536 \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    2>&1 | tee $LOG
