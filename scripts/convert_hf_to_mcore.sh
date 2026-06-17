#!/bin/bash
# ==============================================================================
# convert_hf_to_mcore.sh — HF → Megatron torch_dist 转换
# ==============================================================================
# 调 slime 官方 tools/convert_hf_to_torch_dist.py
# 配合 scripts/models/<name>.sh 使用(slime 内置,直接 source)
#
# 用法:
#   bash convert_hf_to_mcore.sh <model_name>
#   bash convert_hf_to_mcore.sh qwen3-8b
#   bash convert_hf_to_mcore.sh qwen3.5-9b
#   bash convert_hf_to_mcore.sh qwen3.6-35b-a3b
#   bash convert_hf_to_mcore.sh deepseek-v4-flash
#
# 输出: $LZY_ROOT/checkpoints/<model_name>_torch_dist/
# ==============================================================================

set -e
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
source $LZY_ROOT/miniconda3/etc/profile.d/conda.sh
conda activate agentic-opd-train

# 编译器
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LZY_ROOT/shared_libs:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$LZY_ROOT/repos/Megatron-LM:${PYTHONPATH:-}

MODEL_NAME=${1:-qwen3-8b}
SLIME_DIR=$LZY_ROOT/repos/slime
CKPT_ROOT=$LZY_ROOT/checkpoints
mkdir -p $CKPT_ROOT

case $MODEL_NAME in
    qwen3-8b)
        HF_PATH=$LZY_ROOT/models/Qwen3-8B
        SAVE=$CKPT_ROOT/Qwen3-8B_torch_dist
        source $SLIME_DIR/scripts/models/qwen3-8B.sh
        ;;
    qwen3.5-9b|qwen35-9b)
        HF_PATH=$LZY_ROOT/models/Qwen3.5-9B
        SAVE=$CKPT_ROOT/Qwen3.5-9B_torch_dist
        # Qwen3.5 slime 模板(若官方有 qwen3-9B.sh 用之;否则照 qwen3-8B.sh 改)
        if [ -f $SLIME_DIR/scripts/models/qwen3-9B.sh ]; then
            source $SLIME_DIR/scripts/models/qwen3-9B.sh
        else
            echo "WARN: qwen3-9B.sh not in slime; copying qwen3-8B.sh template"
            # Qwen3.5 与 Qwen3 同架构(都是 qwen3 系列),只 vocab/layer 等可能不同
            source $SLIME_DIR/scripts/models/qwen3-8B.sh
        fi
        ;;
    qwen3.6-27b|qwen36-27b)
        HF_PATH=$LZY_ROOT/models/qwen3.6-27B
        SAVE=$CKPT_ROOT/qwen3.6-27B_torch_dist
        source $SLIME_DIR/scripts/models/qwen3-8B.sh  # TODO: 写 qwen3.6-27B.sh
        ;;
    qwen3.6-35b-a3b|qwen36-35b-a3b)
        HF_PATH=$LZY_ROOT/models/qwen3.6-35B-A3B
        SAVE=$CKPT_ROOT/qwen3.6-35B-A3B_torch_dist
        # MoE:照 qwen3-30B-A3B.sh 模板(若 slime 有)
        if [ -f $SLIME_DIR/scripts/models/qwen3-30B-A3B.sh ]; then
            source $SLIME_DIR/scripts/models/qwen3-30B-A3B.sh
        else
            echo "ERROR: qwen3-30B-A3B.sh not found in slime. Need to create."
            exit 1
        fi
        ;;
    deepseek-v4-flash)
        HF_PATH=$LZY_ROOT/models/deepseek-v4-flash
        SAVE=$CKPT_ROOT/deepseek-v4-flash_torch_dist
        # DeepSeek-V4 是 deepseek_v3 的升级,模型结构基本一致;
        # slime quick start 说 kimik2 用 model_type=deepseek_v3,DeepSeek-V4 同理
        if [ -f $SLIME_DIR/scripts/models/deepseek-v3.sh ]; then
            source $SLIME_DIR/scripts/models/deepseek-v3.sh
        elif [ -f $SLIME_DIR/scripts/models/DeepSeek-V3.sh ]; then
            source $SLIME_DIR/scripts/models/DeepSeek-V3.sh
        else
            echo "WARN: deepseek-v3.sh not found in slime scripts/models/"
            echo "Falling back to GLM model template (similar MoE arch)"
            source $SLIME_DIR/scripts/models/glm4-9B.sh 2>/dev/null || {
                echo "ERROR: no suitable model template found"
                exit 1
            }
        fi
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        echo "Usage: $0 {qwen3-8b|qwen3.5-9b|qwen3.6-27b|qwen3.6-35b-a3b|deepseek-v4-flash}"
        exit 1
        ;;
esac

echo "Converting $HF_PATH → $SAVE"
echo "MODEL_ARGS:"
printf '  %s\n' "${MODEL_ARGS[@]}"

python $SLIME_DIR/tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint $HF_PATH \
    --save $SAVE

echo ""
echo "Done. Output:"
ls -la $SAVE
