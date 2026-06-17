#!/bin/bash
# ==============================================================================
# run_smoke_verl_grpo.sh — verl GRPO full-parameter smoke test
# ==============================================================================
# 适配:agentic-opd-verl env (verl 0.7.1 + vllm 0.8.5 + FSDP,无 LoRA)
# 模型:Qwen3-4B-Instruct-2507(新下载,~7.6G)
# 数据:$LZY_ROOT/dataset/standard_rl/(7473 train + test,**已预处理好** GSM8K 格式)
# GPU:2(FSDP 不切 TP;1 train worker + 1 rollout vllm worker)
# 时间:~30-60 min(2 epoch,几百个 sample)
# 用途:验证 verl + FSDP + vllm + GRPO 端到端管线
#
# 用法(GPU 实例):
#   source $LZY_ROOT/activate_lzy.sh
#   conda activate agentic-opd-verl
#   bash $LZY_ROOT/projects/agentic-opd/scripts/run_smoke_verl_grpo.sh
# ==============================================================================

set -e

export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy

MODEL_PATH=$LZY_ROOT/models/Qwen3-4B-Instruct-2507
TRAIN_DATA=$LZY_ROOT/dataset/standard_rl/train.parquet
VAL_DATA=$LZY_ROOT/dataset/standard_rl/test.parquet

# ---------- 0. 路径验证 ----------
if [ ! -d "$MODEL_PATH" ]; then
    echo "[FATAL] model not found: $MODEL_PATH"
    echo "        请先下载 Qwen3-4B-Instruct-2507 到 $LZY_ROOT/models/"
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

# ---------- 1. 启动 verl GRPO ----------
echo "=========================================="
echo "verl GRPO smoke test"
echo "Model: $MODEL_PATH"
echo "Train: $TRAIN_DATA"
echo "Val:   $VAL_DATA"
echo "GPU:   2 (n_gpus_per_node=2)"
echo "=========================================="

set -x

# 关键:不切 LoRA(actor_rollout_ref.model.lora_rank=0)
# 全参 FSDP:1 actor + 1 ref + 1 rollout(vllm),共 2 GPU
# 训练侧 GPU 0,rollout 侧 GPU 1(via tensor_model_parallel_size=1)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_smoke' \
    trainer.experiment_name='qwen3_4b_instruct_grpo_smoke' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "verl GRPO smoke test exit code: $EXIT_CODE"
echo "  0 = success (pipeline ran end-to-end)"
echo "  non-zero = failure (check log)"
echo "=========================================="
exit $EXIT_CODE
