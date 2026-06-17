#!/usr/bin/env bash
set -euo pipefail

export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy

source "$LZY_ROOT/activate_lzy.sh"
verl128

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export CUDA_MODULE_LOADING=LAZY
export VLLM_USE_V1=1

MODEL_PATH=$LZY_ROOT/models/Qwen3-4B-Instruct-2507
TRAIN_DATA=$LZY_ROOT/dataset/standard_rl/train.parquet
VAL_DATA=$LZY_ROOT/dataset/standard_rl/test.parquet

if [ ! -d "$MODEL_PATH" ]; then
    echo "[FATAL] model not found: $MODEL_PATH"
    echo "        If you did not create smoke-sdpa copy, change MODEL_PATH to original model and set use_remove_padding=False."
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[FATAL] train data not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "[FATAL] val data not found: $VAL_DATA"
    exit 1
fi

echo "=========================================="
echo "verl128 GRPO smoke test"
echo "Model: $MODEL_PATH"
echo "Train: $TRAIN_DATA"
echo "Val:   $VAL_DATA"
echo "=========================================="

echo "== env check =="
which python
python - <<'PY'
import os
import torch
import vllm
import verl

print("CONDA_PREFIX", os.environ.get("CONDA_PREFIX"))
print("python", os.sys.executable)
print("torch", torch.__version__, torch.version.cuda)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
print("vllm", getattr(vllm, "__version__", "unknown"))
print("verl file", verl.__file__)
PY

cd "$LZY_ROOT/repos/verl-cu128-vllm"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_smoke' \
    trainer.experiment_name='qwen3_4b_instruct_grpo_1step_verl128' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    "$@"
