#!/bin/bash
# ==============================================================================
# run_slime_opd_qwen35_9b.sh — slime OPD with Qwen3.5-9B student
# ==============================================================================
# 你 wiki 里 RLSD/OPSD 主线 + Qwen3.5-9B multimodal
# teacher: DeepSeek-V4-Flash(sglang server,port 30000)
#
# 前置:
#   1. bash scripts/start_ds_v4_flash.sh 30000 &   # teacher
#   2. bash scripts/convert_hf_to_mcore.sh qwen3.5-9b
#   3. 数据集准备好
#
# 用法:
#   bash run_slime_opd_qwen35_9b.sh
# ==============================================================================

set -e
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
source $LZY_ROOT/miniconda3/etc/profile.d/conda.sh
conda activate agentic-opd-train

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LZY_ROOT/shared_libs:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$LZY_ROOT/repos/Megatron-LM:$LZY_ROOT/repos/slime:${PYTHONPATH:-}
export HF_HOME=$LZY_ROOT/.cache/huggingface
export TRITON_CACHE_DIR=$LZY_ROOT/.cache/triton

SLIME_DIR=$LZY_ROOT/repos/slime

# === 预检 ===
echo "Pre-flight checks"

# Qwen3.5 转换(Qwen3.5 与 Qwen3 同架构,沿用 qwen3-8B.sh 模板)
if [ ! -d $LZY_ROOT/checkpoints/Qwen3.5-9B_torch_dist ]; then
    echo "Converting Qwen3.5-9B to mcore..."
    bash $LZY_ROOT/projects/agentic-opd/scripts/convert_hf_to_mcore.sh qwen3.5-9b
fi

DATA=$LZY_ROOT/datasets/dapo-math-17k/dapo-math-17k.jsonl
if [ ! -f $DATA ]; then
    echo "ERROR: $DATA not found"
    echo "  跑: huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $LZY_ROOT/datasets/dapo-math-17k"
    exit 1
fi

# teacher
if ! curl -s -m 3 http://localhost:30000/v1/models > /dev/null 2>&1; then
    echo "ERROR: teacher server not responding on :30000"
    echo "  跑: bash $LZY_ROOT/projects/agentic-opd/scripts/start_ds_v4_flash.sh 30000 &"
    exit 1
fi

# === 启动 ===
# slime 的 examples/on_policy_distillation/ 默认 example 是 Qwen3-8B,
# Qwen3.5-9B 用同一个 model family,改 4 行即可
#
# slime 的 opd 例启动逻辑(在 run-qwen3-8B-opd.sh 里):
#   1. 起 SGLang teacher server(--opd-type sglang)  ← 我们已经手动起了
#   2. ray start --head
#   3. python train.py --use-opd --opd-type sglang --rm-url http://localhost:30000
#   4. 训练循环

# 直接调 slime 的 train.py(参考 example 启动)
export HF_CHECKPOINT=$LZY_ROOT/models/Qwen3.5-9B
export REF_MODEL_PATH=$LZY_ROOT/checkpoints/Qwen3.5-9B_torch_dist
export PROMPT_DATA=$DATA
export RM_URL=http://localhost:30000
export OPD_TYPE=sglang

LOG=$LZY_ROOT/logs/agentic-opd/opd_qwen35_9b_$(date +%Y%m%d_%H%M%S).log
mkdir -p $(dirname $LOG)

echo ""
echo "=========================================="
echo "Starting slime OPD (Qwen3.5-9B + DeepSeek-V4-Flash teacher)"
echo "  student: $HF_CHECKPOINT"
echo "  teacher: $RM_URL"
echo "  log: $LOG"
echo "=========================================="
echo ""

# ray 启动
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-gpus 8 2>&1 | head -5

# 跑 slime OPD
cd $SLIME_DIR
python train.py \
    --use-opd \
    --opd-type sglang \
    --rm-url $RM_URL \
    --actor-model-path $HF_CHECKPOINT \
    --ref-model-path $REF_MODEL_PATH \
    --prompt-data $PROMPT_DATA \
    --apply-chat-template \
    --rollout-batch-size 16 \
    --n-samples-per-prompt 8 \
    --num-rollout 100 \
    --global-batch-size 128 \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-response-length 4096 \
    --context-length 16384 \
    --learning-rate 1e-6 \
    --clip-eps-low 0.2 --clip-eps-high 0.28 \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --sglang-tp-size 8 \
    --save-interval 20 \
    --save-path $LZY_ROOT/output/agentic-opd/opd_qwen35_9b_$(date +%Y%m%d_%H%M%S) \
    2>&1 | tee $LOG
