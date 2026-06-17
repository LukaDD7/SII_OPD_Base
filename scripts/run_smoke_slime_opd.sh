#!/bin/bash
# ==============================================================================
# run_smoke_slime_opd.sh — slime OPD self-distillation smoke test
# ==============================================================================
# 适配:agentic-opd-train env (slime 0.3.0 + sglang 0.5.12.post1 + Megatron)
# 模型:Qwen3-8B 当 student,Qwen3-8B 当 dummy teacher(自蒸馏,只为跑通管线)
# 数据:dapo-math-17k
# GPU:1 teacher + 1 actor + 1 rollout = 3 GPU(原 recipe 是 7 GPU,缩 70%)
# 时间:~15-30 min(5 rollouts)
# 用途:验证 slime OPD 端到端管线(teacher server → student rollout → loss → step)
#
# 用法(GPU 实例):
#   source $LZY_ROOT/activate_lzy.sh
#   conda activate agentic-opd-train
#   bash $LZY_ROOT/projects/agentic-opd/scripts/run_smoke_slime_opd.sh
# ==============================================================================

set -e

# ---------- 1. 路径常量 ----------
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
export SLIME_DIR=$LZY_ROOT/repos/slime
export MEGATRON_DIR=$LZY_ROOT/repos/Megatron-LM
export SGLANG_DIR=$LZY_ROOT/repos/sglang

MODEL_DIR=$LZY_ROOT/models/Qwen3-8B
DATASET_PATH=$LZY_ROOT/dataset/dapo-math-17k/dapo-math-17k.jsonl
MCORE_DIR=$LZY_ROOT/checkpoints/Qwen3-8B_torch_dist_smoke
SAVE_DIR=$LZY_ROOT/checkpoints/Qwen3-8B_slime_smoke

# ---------- 2. 启动 teacher server(Qwen3-8B sglang server) ----------
TEACHER_IP="127.0.0.1"
TEACHER_PORT=13141
TEACHER_GPU=0     # teacher 跑在 GPU 0
LOG_FILE="/tmp/sglang_teacher_$(date +%s).log"

echo "[STEP 1] Launching teacher sglang server on GPU $TEACHER_GPU..."
CUDA_VISIBLE_DEVICES=$TEACHER_GPU python3 -m sglang.launch_server \
    --model-path $MODEL_DIR \
    --host 0.0.0.0 \
    --port $TEACHER_PORT \
    --tp 1 \
    --chunked-prefill-size 4096 \
    --mem-fraction-static 0.4 \
    --context-length 8192 \
    --attention-backend triton \
    > "$LOG_FILE" 2>&1 &

TEACHER_PID=$!
echo "  teacher sglang server PID: $TEACHER_PID"
echo "  log: $LOG_FILE"

# 等 teacher server ready
echo "[STEP 2] Waiting for teacher server to be ready..."
for i in {1..120}; do
    if curl -sf http://$TEACHER_IP:$TEACHER_PORT/health_generate > /dev/null 2>&1; then
        echo "  teacher server READY after ${i} attempts"
        curl -s http://$TEACHER_IP:$TEACHER_PORT/get_model_info
        echo ""
        break
    fi
    if ! kill -0 $TEACHER_PID 2>/dev/null; then
        echo "  [FATAL] teacher sglang server died. Last 30 lines of log:"
        tail -30 "$LOG_FILE"
        exit 1
    fi
    sleep 5
done

# 再等 10s 稳定
sleep 10

# ---------- 3. 把 HF 权重转 mcore(slime 的 trainer 用 Megatron 格式)----------
if [ ! -d "$MCORE_DIR" ] || [ -z "$(ls -A $MCORE_DIR 2>/dev/null)" ]; then
    echo "[STEP 3] Converting HF → Megatron torch_dist (1 GPU)..."
    source $SLIME_DIR/scripts/models/qwen3-8B.sh
    export PYTHONPATH=$MEGATRON_DIR:$SGLANG_DIR/python:$PYTHONPATH
    CUDA_VISIBLE_DEVICES=1 python3 $SLIME_DIR/tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint $MODEL_DIR \
        --save $MCORE_DIR \
        2>&1 | tail -20
    echo "  mcore convert DONE"
else
    echo "[STEP 3] mcore dir exists, skip convert: $MCORE_DIR"
fi

# ---------- 4. 启动 ray + 跑 slime OPD ----------
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=$MEGATRON_DIR:$SGLANG_DIR/python:$SLIME_DIR:${PYTHONPATH:-}

# 启动 ray head(num-gpus 4 覆盖 actor+rollout+teacher 共 3 卡 + 1 备用)
echo "[STEP 4] Starting ray head..."
ray start --head --node-ip-address 127.0.0.1 --num-gpus 4 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

# 等 ray ready
sleep 5
ray status

source $SLIME_DIR/scripts/models/qwen3-8B.sh

# 关键 args(缩到最小)
CKPT_ARGS=(
    --hf-checkpoint $MODEL_DIR
    --ref-load $MCORE_DIR
    --load $SAVE_DIR
    --save $SAVE_DIR
    --save-interval 100
)

ROLLOUT_ARGS=(
    --prompt-data $DATASET_PATH
    --input-key prompt
    --apply-chat-template
    --rollout-shuffle
    --num-rollout 5
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --rollout-max-response-len 2048
    --rollout-temperature 1
    --global-batch-size 4
    --balance-data
)

RM_ARGS=(
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
    --rm-url http://$TEACHER_IP:$TEACHER_PORT/generate
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type sglang
    --opd-kl-coef 1.0
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
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
    --sglang-mem-fraction-static 0.4
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

echo "[STEP 5] Submitting slime OPD job to ray..."
# actor GPU 1 + rollout GPU 2(GPU 0 是 teacher)
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{
      "env_vars": {
        "PYTHONPATH": "'$MEGATRON_DIR':'$SGLANG_DIR'/python:'$SLIME_DIR'",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
      }
    }' \
    -- python3 $SLIME_DIR/train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-num-gpus 1 \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${RM_ARGS[@]}

EXIT_CODE=$?

echo ""
echo "[CLEANUP] Killing teacher server and ray..."
kill $TEACHER_PID 2>/dev/null
pkill -9 sglang 2>/dev/null
sleep 3
ray stop --force 2>/dev/null
pkill -9 ray 2>/dev/null
pkill -9 python 2>/dev/null

echo ""
echo "=========================================="
echo "smoke test exit code: $EXIT_CODE"
echo "  0 = success (pipeline ran end-to-end)"
echo "  non-zero = failure (check log)"
echo "=========================================="
exit $EXIT_CODE
