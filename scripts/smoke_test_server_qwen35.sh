#!/bin/bash
# ==============================================================================
# smoke_test_server_qwen35.sh — GPU 实例 Qwen3.5-9B sglang server 验证
# ==============================================================================
# 在 GPU 实例跑,1 卡 Qwen3.5-9B sglang server
# 验证:1) sglang 启动;2) /v1/models 返回;3) /v1/chat/completions 推理
#
# 前置: env 装好 + nvidia libs 装好 + 有 GPU
# 用法:
#   bash smoke_test_server_qwen35.sh
# ==============================================================================

set -e
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
source $LZY_ROOT/miniconda3/etc/profile.d/conda.sh
conda activate agentic-opd-train

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LZY_ROOT/shared_libs:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=$LZY_ROOT/.cache/huggingface
export TRITON_CACHE_DIR=$LZY_ROOT/.cache/triton
export SGLANG_TP_SIZE=1
export SGLANG_KV_CACHE_MEM_FRACTION=0.85
export SGLANG_USE_DEEP_GEMM=1

PORT=${1:-30099}
LOG=/tmp/sglang_qwen35_smoke.log

# 1. 检查 GPU
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>&1)
echo "Detected GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" = "0" ]; then
    echo "ERROR: no CUDA GPU detected. This smoke test needs GPU."
    exit 1
fi

# 2. 起 server(1 卡)
echo "Starting sglang server (Qwen3.5-9B, port=$PORT)..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path $LZY_ROOT/models/Qwen3.5-9B \
    --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size 1 \
    --mem-fraction-static 0.85 \
    --context-length 16384 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    > $LOG 2>&1 &

SERVER_PID=$!
echo "Server PID=$SERVER_PID, log=$LOG"

# 3. 等 server ready(最多 180s)
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s -m 2 http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
        echo "[OK] Server ready after ${i}*3s"
        break
    fi
    sleep 3
done

# 4. 测试 /v1/models
echo ""
echo "--- /v1/models ---"
curl -s -m 5 http://127.0.0.1:$PORT/v1/models | python -m json.tool 2>&1 | head -10

# 5. 测试简单 chat
echo ""
echo "--- /v1/chat/completions: 1+1=? ---"
RESP=$(curl -s -m 30 -X POST http://127.0.0.1:$PORT/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen3.5-9B",
        "messages": [{"role": "user", "content": "1+1=?"}],
        "max_tokens": 16,
        "temperature": 0
    }')
echo "$RESP" | python -m json.tool 2>&1 | head -20
ANSWER=$(echo "$RESP" | python -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
if [[ "$ANSWER" == *"2"* ]]; then
    echo "[OK] Answer contains '2'"
else
    echo "[WARN] Answer: $ANSWER (expected to contain '2')"
fi

# 6. 收尾
echo ""
echo "Stopping server (PID=$SERVER_PID)..."
kill -TERM $SERVER_PID 2>/dev/null || true
sleep 2
pkill -f "sglang.launch_server" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Smoke test done. Full log: $LOG"
echo "=========================================="
