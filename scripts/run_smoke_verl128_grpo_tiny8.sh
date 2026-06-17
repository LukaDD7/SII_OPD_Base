#!/usr/bin/env bash
set -euo pipefail

export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy

source "$LZY_ROOT/activate_lzy.sh"
verl128

# Use logical CUDA IDs inside container, not /dev/nvidia minor numbers.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}

# Stable vLLM/verl smoke settings.
export VLLM_USE_V1=0
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_MODULE_LOADING=LAZY
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export VERL_SMOKE_DUMMY_REWARD=1

MODEL_PATH="$LZY_ROOT/models/Qwen3-4B-Instruct-2507"
TRAIN_DATA="$LZY_ROOT/dataset/standard_rl/train_tiny8.parquet"
VAL_DATA="$LZY_ROOT/dataset/standard_rl/test_tiny8.parquet"

ray stop -f || true

echo "[smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[smoke] MODEL_PATH=$MODEL_PATH"
echo "[smoke] TRAIN_DATA=$TRAIN_DATA"
echo "[smoke] VAL_DATA=$VAL_DATA"

python - <<'PY'
import os
import torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch", torch.__version__, torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

cd "$LZY_ROOT/repos/verl-cu128-vllm"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train=False \
  trainer.critic_warmup=0 \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  'trainer.logger=["console"]' \
  trainer.project_name=verl_grpo_smoke \
  trainer.experiment_name=qwen3_4b_instruct_grpo_tiny8_verl128 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1
