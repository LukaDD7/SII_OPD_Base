# Smoke Tests — slime OPD + verl GRPO

> 创建日期:2026-06-10
> 状态:CPU 端脚本已写好、syntax verified;待 GPU 实例跑通
> 用途:在 GPU 实例上**低开销**验证两条研究路线(slime + verl)的端到端管线

---

## 0. 准备工作(只需做一次)

### 0.1 验证下载和 env(CPU 实例)

```bash
# 确认下载的 3 个东西
ls /inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/dapo-math-17k/dapo-math-17k.jsonl  # slime OPD 数据
ls /inspire/hdd/global_user/mengweicheng-240108120092/lzy/dataset/standard_rl/                     # verl GRPO 数据(已存在)
ls /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-4B-Instruct-2507/           # verl 模型
ls /inspire/hdd/global_user/mengweicheng-240108120092/lzy/models/Qwen3-VL-8B-Instruct/             # VLM 模型(待用)

# 确认两个 env 装好
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/agentic-opd-train/smoke_test_imports.sh  # 24/24
bash /inspire/hdd/global_user/mengweicheng-240108120092/lzy/envs/agentic-opd-verl/smoke_test_imports.sh   # 24/24
```

### 0.2 GPU 实例 baseline check(SII skill)

```bash
# 在 GPU 实例跑,SII skill 的 L1-L5 audit
python3 /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/shared-agent-skills/SII_GPU_env_construct/scripts/check_gpu_baseline.py \
    --env-name agentic-opd-train --phase gpu

python3 /inspire/hdd/global_user/mengweicheng-240108120092/lzy/projects/shared-agent-skills/SII_GPU_env_construct/scripts/check_gpu_baseline.py \
    --env-name agentic-opd-verl --phase gpu
```

期望:L1-L5 全 OK(尤其 L4 `torch.cuda.is_available()=True` + L5 `/dev/nvidia*` 存在)

---

## 1. slime OPD 自蒸馏 smoke

### 1.1 配置

| 维度 | 值 |
|------|-----|
| Env | `agentic-opd-train` |
| 训练后端 | Megatron(slime 强制) |
| 推理后端 | sglang 0.5.12.post1 |
| Student | Qwen3-8B(本地已有) |
| Teacher | Qwen3-8B(自蒸馏,dummy teacher) |
| 数据集 | `dapo-math-17k`(~17k prompts,只用 5 个 rollout) |
| GPU | 1 teacher + 1 actor + 1 rollout = **3 GPU** |
| 时间 | ~15-30 min |
| 预期结果 | 5 个 rollout 都成功 + trainer step loss 正常打印 |

### 1.2 跑(GPU 实例)

```bash
source $LZY_ROOT/activate_lzy.sh
opd   # 激活 slime env
bash $LZY_ROOT/projects/agentic-opd/scripts/run_smoke_slime_opd.sh
```

### 1.3 跑成功的标志

- 看到 `[STEP 2] teacher server READY after N attempts`(N < 30 算正常)
- 看到 `[STEP 3] mcore convert DONE`
- 看到 ray status 显示 3 个 GPU 都被分配
- slime job 跑完没 crash,日志里看到:
  - `loss: 5.x` → `loss: 4.x`(在下降)
  - `mean_reward: 0.x`(可能不理想,因为 dummy teacher)
  - `kl: 0.0x`(KL 应该很小,因为 student ≈ teacher)
- 退出码 = 0

### 1.4 跑失败的常见原因

| 错误 | 原因 | 修复 |
|------|------|------|
| teacher server 启动后立刻死 | sglang kernel 与 torch 版本不匹配 | 重新 `pip install sglang==0.5.12.post1` |
| `CUDA out of memory` | Qwen3-8B 单卡不够 | 把 `--mem-fraction-static` 从 0.4 降到 0.3,或扩到 5 GPU |
| `torch.distributed.init_process_group` 失败 | ray + GPU 通信问题 | `ray stop --force` 重试 |
| mcore convert 卡住 | 缺 `rotary_base` 等参数 | 检查 `slime/scripts/models/qwen3-8B.sh` 的 rotary-base 是否与 HF config.json 一致 |

---

## 2. verl GRPO 全参 smoke

### 2.1 配置

| 维度 | 值 |
|------|-----|
| Env | `agentic-opd-verl` |
| 训练后端 | FSDP |
| 推理后端 | vLLM 0.8.5 |
| Student | Qwen3-4B-Instruct-2507(新下载,~7.6G) |
| Teacher | 无(纯 GRPO,无 OPD) |
| 数据集 | `standard_rl/train.parquet`(7473 prompts,**已预处理好**) |
| GPU | **2**(1 actor + 1 vllm rollout) |
| 时间 | ~30-60 min(2 epoch) |
| 预期结果 | 2 epoch 跑完,reward/kl/entropy 都正常打印 |

### 2.2 跑(GPU 实例)

```bash
source $LZY_ROOT/activate_lzy.sh
opd-verl   # 激活 verl env
bash $LZY_ROOT/projects/agentic-opd/scripts/run_smoke_verl_grpo.sh
```

### 2.3 跑成功的标志

- Ray cluster 启动成功
- actor worker 在 GPU 0,vllm worker 在 GPU 1
- 日志里看到:
  - `actor_loss: x.xx` 在打印
  - `val_reward: x.xx`(可能很低,因为 2 epoch 不够训)
  - `kl: 0.0xx`
  - 训完 2 epoch 退出
- 退出码 = 0

### 2.4 跑失败的常见原因

| 错误 | 原因 | 修复 |
|------|------|------|
| `vllm._C` import 失败 | libcuda.so.1 路径问题 | `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` |
| `OOM` 在 actor 训练时 | FSDP 没 offload | 把 `actor_rollout_ref.actor.fsdp_config.optimizer_offload=True` |
| `vllm EngineCore failed` | GPU 1 被其他进程占 | `nvidia-smi` 看谁占 GPU 1,杀 |
| 数据集 `prompt` 字段格式不对 | standard_rl 是裸字符串 prompt,verl 要 chat template | 在 `data.apply_chat_template=True` 加上 |

---

## 3. 与完整训练的对比

| 维度 | smoke test | 完整训练 |
|------|-----------|----------|
| 数据量 | 5-100 prompts | 全量(17k / 7.5k) |
| Rollout/Epoch | 5 / 2 | 300 / 15-30 |
| Response length | 2k-4k tokens | 16k tokens |
| 蒸馏目标 | dummy teacher / 无 | 真 teacher(DeepSeek-V4-Flash) |
| GPU | 2-3 | 8(满 H200) |
| 时间 | 15-60 min | 几天 |

**smoke test 不是为了训出好模型,是为了验证管线不挂**。

---

## 4. 之后:从 smoke 扩到完整训练

### 4.1 slime OPD → 完整

1. 改 `run_smoke_slime_opd.sh`:
   - 改 teacher 路径为 `$LZY_ROOT/models/deepseek-v4-flash`(真 teacher)
   - `--num-rollout 300 --rollout-batch-size 16 --n-samples-per-prompt 4`(原 recipe 默认)
   - `--global-batch-size 64`(原 recipe)
   - GPU 扩到 1 teacher (TP=4 EP=4) + 7 student

2. 处理 teacher:
   - deepseek-v4-flash 是 MoE,`sglang.launch_server` 需要 `--tensor-parallel-size 4 --expert-parallel-size 4`
   - 若 sglang 0.5.12.post1 不识别 `deepseek_v4` model_type,改 config.json `model_type: "deepseek_v3"`(同 kimik2 trick)

### 4.2 verl GRPO → verl OPD

需要写 plugin(目前 verl 没原生 OPD):
1. `verl/trainer/ppo/core_algos.py` 加 `compute_distillation_loss`
2. 改 trainer config,加 `--teacher-endpoint` 参数
3. 起 teacher sglang server(reuse `start_ds_v4_flash.sh`)

详细步骤等 v6 文档。

---

## 5. 引用

- slime 官方 OPD recipe:`$SLIME_DIR/examples/on_policy_distillation/run-qwen3-8B-opd.sh`
- verl 官方 GRPO recipe:`$VERL_DIR/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`
- SII_GPU_env_construct skill:`$LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/SKILL.md`
- agentic-opd-config.md(infra 总规范):`$LZY_ROOT/docs/agentic-opd-config.md`
