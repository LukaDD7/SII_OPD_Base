#!/bin/bash
# ==============================================================================
# run_slime_grpo_qwen3_8b.sh — slime GRPO baseline(对照)
# ==============================================================================
# 用 slime examples/run-glm4-9B.sh 或 geo3k_vlm 模板做 GRPO
# 这是 GRPO baseline,作为 OPD 的对照
#
# 前置:与 run_slime_opd_qwen3_8b.sh 相同
# 用法:
#   bash run_slime_grpo_qwen3_8b.sh
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

# === 1. 预检 ===
echo "Pre-flight checks"

if [ ! -d $LZY_ROOT/checkpoints/Qwen3-8B_torch_dist ]; then
    echo "Converting Qwen3-8B to mcore..."
    bash $LZY_ROOT/projects/agentic-opd/scripts/convert_hf_to_mcore.sh qwen3-8b
fi

DATA=$LZY_ROOT/datasets/dapo-math-17k/dapo-math-17k.jsonl
if [ ! -f $DATA ]; then
    echo "ERROR: $DATA not found"
    echo "  huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir $LZY_ROOT/datasets/dapo-math-17k"
    exit 1
fi

# === 2. slime GRPO 启动 ===
# slime GRPO 模板:examples/geo3k_vlm/ 或 run-glm4-9B.sh
# 这里给一个自定义 GRPO 启动(用 Qwen3-8B base)
export HF_CHECKPOINT=$LZY_ROOT/models/Qwen3-8B
export REF_MODEL_PATH=$LZY_ROOT/checkpoints/Qwen3-8B_torch_dist
export PROMPT_DATA=$DATA
export SAVE_DIR=$LZY_ROOT/output/agentic-opd/grpo_qwen3_8b_$(date +%Y%m%d_%H%M%S)
mkdir -p $SAVE_DIR

# 8 卡 GRPO 配置
# 注:slime GRPO 详细配置参考 $SLIME_DIR/examples/geo3k_vlm/ 的启动脚本
echo ""
echo "GRPO config (Qwen3-8B, 8×H200):"
echo "  student: $HF_CHECKPOINT"
echo "  ref mcore: $REF_MODEL_PATH"
echo "  save: $SAVE_DIR"
echo ""

cat > $LZY_ROOT/projects/agentic-opd/configs/slime/grpo_qwen3_8b.yaml <<EOF
# GRPO baseline for Qwen3-8B
# 调 slime train.py(参考 slime/ray_train.py 或 slime/entrypoints/)
rollout:
  model_path: $HF_CHECKPOINT
  ref_model_path: $REF_MODEL_PATH
  prompt_data: $PROMPT_DATA
  n_samples_per_prompt: 8
  rollout_batch_size: 32
  temperature: 1.0
  top_p: 0.95
  max_response_length: 4096
  context_length: 8192
  tensor_parallel_size: 8
  sglang_server_args:
    mem_fraction_static: 0.82

train:
  global_batch_size: 64
  num_steps_per_rollout: 1
  num_rollout: 100
  learning_rate: 1.0e-6
  clip_eps_low: 0.2
  clip_eps_high: 0.28

save:
  save_interval: 20
  save_path: $SAVE_DIR
EOF

echo "Config written to: $LZY_ROOT/projects/agentic-opd/configs/slime/grpo_qwen3_8b.yaml"
echo ""
echo "To run GRPO, use slime entry point:"
echo "  cd $SLIME_DIR"
echo "  # 参考 examples/geo3k_vlm/run.sh 把配置改成上面的 yaml"
echo ""
echo "Or for a single-step smoke test:"
echo "  cd $SLIME_DIR && python -m slime.train --config $LZY_ROOT/projects/agentic-opd/configs/slime/grpo_qwen3_8b.yaml --num-steps 1"
