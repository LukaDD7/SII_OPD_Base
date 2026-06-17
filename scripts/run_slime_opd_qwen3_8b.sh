#!/bin/bash
# ==============================================================================
# run_slime_opd_qwen3_8b.sh — slime OPD 跑通测试
# ==============================================================================
# 复用 slime 官方 examples/on_policy_distillation/run-qwen3-8B-opd.sh
# 这是你 wiki 里 OPSD/RLSD 主线对应的官方实现
#
# 前置:
#   1. bash scripts/start_ds_v4_flash.sh 30000   # 起 teacher
#   2. bash scripts/convert_hf_to_mcore.sh qwen3-8b  # 转换 mcore
#   3. 数据集已下载到 $LZY_ROOT/datasets/dapo-math-17k
#
# 用法:
#   bash run_slime_opd_qwen3_8b.sh           # 跑 OPD
#   bash run_slime_opd_qwen3_8b.sh --dryrun  # 不真跑,只检查环境
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
EXAMPLE=$SLIME_DIR/examples/on_policy_distillation/run-qwen3-8B-opd.sh

# === 1. 预检 ===
echo "============================================"
echo "Pre-flight checks"
echo "============================================"

# 1a. 官方 example 存在?
if [ ! -f $EXAMPLE ]; then
    echo "ERROR: $EXAMPLE not found"
    echo "  slimerl/slime 没 clone 完? 跑 git clone"
    exit 1
fi
echo "[OK] slime example exists: $EXAMPLE"

# 1b. 转换产物
if [ ! -d $LZY_ROOT/checkpoints/Qwen3-8B_torch_dist ]; then
    echo "WARN: Qwen3-8B_torch_dist not found"
    echo "  跑: bash scripts/convert_hf_to_mcore.sh qwen3-8b"
    if [ "${1}" != "--skip-convert" ]; then
        bash $LZY_ROOT/projects/agentic-opd/scripts/convert_hf_to_mcore.sh qwen3-8b
    fi
fi

# 1c. 数据集
DATA=$LZY_ROOT/datasets/dapo-math-17k/dapo-math-17k.jsonl
if [ ! -f $DATA ]; then
    echo "WARN: $DATA not found"
    echo "  跑: huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $LZY_ROOT/datasets/dapo-math-17k"
fi

# 1d. teacher server
if ! curl -s -m 3 http://localhost:30000/v1/models > /dev/null 2>&1; then
    echo "WARN: teacher server not responding on :30000"
    echo "  跑: bash scripts/start_ds_v4_flash.sh 30000 &"
fi

if [ "${1}" = "--dryrun" ]; then
    echo "[DRYRUN] 预检完成,未启动训练"
    exit 0
fi

# === 2. 改官方 example 的 4 个变量,跑 ===
# 官方 example 默认有这些 export(若没有会因找不到路径 fail),我们用 env 覆盖
export HF_CHECKPOINT=$LZY_ROOT/models/Qwen3-8B
export REF_MODEL_PATH=$LZY_ROOT/checkpoints/Qwen3-8B_torch_dist
export PROMPT_DATA=$DATA
export RM_URL=http://localhost:30000  # DeepSeek-V4-Flash sglang server
export OPD_TYPE=sglang  # 用外部 teacher server

echo ""
echo "============================================"
echo "Starting slime OPD"
echo "  student: $HF_CHECKPOINT"
echo "  teacher: $RM_URL (DeepSeek-V4-Flash)"
echo "  mcore ref: $REF_MODEL_PATH"
echo "  data: $PROMPT_DATA"
echo "============================================"
echo ""

cd $SLIME_DIR
bash $EXAMPLE 2>&1 | tee $LZY_ROOT/logs/agentic-opd/opd_$(date +%Y%m%d_%H%M%S).log
