#!/bin/bash
# ==============================================================================
# eval/eval_harness.sh — 模型评测入口
# ==============================================================================
# 1) 启动 sglang server(给定模型)
# 2) 跑评测(lmsys/lm-eval-harness 或 自定义)
#
# 用法:
#   bash eval/eval_harness.sh qwen35-9b mmlu
#   bash eval/eval_harness.sh deepseek-v4-flash livecodebench
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

MODEL_ALIAS=${1:-qwen35-9b}
BENCHMARK=${2:-mmlu}
PORT=${3:-30100}
LOG_DIR=$LZY_ROOT/logs/agentic-opd/eval
mkdir -p $LOG_DIR

# 模型 → 路径
case $MODEL_ALIAS in
    qwen3-8b)        MODEL_PATH=$LZY_ROOT/models/Qwen3-8B;        TP=1 ;;
    qwen35-9b|qwen3.5-9b)  MODEL_PATH=$LZY_ROOT/models/Qwen3.5-9B;   TP=1 ;;
    qwen36-27b)      MODEL_PATH=$LZY_ROOT/models/qwen3.6-27B;     TP=4 ;;
    qwen36-35b-a3b)  MODEL_PATH=$LZY_ROOT/models/qwen3.6-35B-A3B; TP=4; EP=8 ;;
    deepseek-v4-flash) MODEL_PATH=$LZY_ROOT/models/deepseek-v4-flash; TP=4; EP=4 ;;
    *) echo "Unknown model: $MODEL_ALIAS"; exit 1 ;;
esac

echo "=========================================="
echo "Eval harness"
echo "  model: $MODEL_ALIAS ($MODEL_PATH)"
echo "  benchmark: $BENCHMARK"
echo "  port: $PORT"
echo "=========================================="

# 1. 起 server
LOG=$LOG_DIR/sglang_${MODEL_ALIAS}_$(date +%Y%m%d_%H%M%S).log
echo "Starting sglang server: $LOG"

export SGLANG_TP_SIZE=$TP
[ -n "$EP" ] && export SGLANG_EP_SIZE=$EP
export SGLANG_KV_CACHE_MEM_FRACTION=0.82
export SGLANG_USE_DEEP_GEMM=1

python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port $PORT \
    --tensor-parallel-size $TP \
    --mem-fraction-static 0.82 \
    --context-length 32768 \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    > $LOG 2>&1 &

SERVER_PID=$!
trap "kill -TERM $SERVER_PID 2>/dev/null" EXIT

# 等 server
echo "Waiting for server..."
for i in {1..90}; do
    if curl -s -m 2 http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "[OK] Server ready after ${i}*2s"
        break
    fi
    sleep 2
done

# 2. 跑评测(lmsys/lm-eval-harness,需要单独装)
echo ""
echo "Running $BENCHMARK on $MODEL_ALIAS at port $PORT"
echo ""

case $BENCHMARK in
    mmlu)
        python -m lm_eval --model openai-completions \
            --model_args model=$MODEL_ALIAS,base_url=http://localhost:$PORT/v1,num_concurrent=8 \
            --tasks mmlu_pro,mmlu \
            --batch_size 8 \
            --output_path $LZY_ROOT/output/agentic-opd/eval/$BENCHMARK-$MODEL_ALIAS
        ;;
    livecodebench)
        # 简化的代码生成评测,用 HumanEval 替代
        python -m lm_eval --model openai-completions \
            --model_args model=$MODEL_ALIAS,base_url=http://localhost:$PORT/v1,num_concurrent=4 \
            --tasks humaneval \
            --batch_size 4 \
            --output_path $LZY_ROOT/output/agentic-opd/eval/$BENCHMARK-$MODEL_ALIAS
        ;;
    *)
        echo "TODO: implement $BENCHMARK (use lm-eval-harness as template)"
        ;;
esac

# 3. 收尾
echo ""
echo "Stopping server (PID=$SERVER_PID)..."
kill -TERM $SERVER_PID 2>/dev/null
sleep 3
pkill -f "sglang.launch_server" 2>/dev/null

echo ""
echo "=========================================="
echo "Eval done. Output: $LZY_ROOT/output/agentic-opd/eval/$BENCHMARK-$MODEL_ALIAS"
echo "=========================================="
