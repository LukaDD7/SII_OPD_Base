# Agentic On-Policy Distillation — 训练 + 推理 Infra 项目

> 创建:2026-06-01
> 状态:env 重建中(详见 `docs/agentic-opd-config-v6-cu128.md`,**CUDA 12.8 baseline**)
> 目的:**搭好底座,新算法 → 改 plugin + 改 config 即可跑**
> 硬件:8 × H200 141GB(单节点,driver 主流 570.x → 限 cu128)

## 目录结构

```
agentic-opd/
├── CLAUDE.md                        # 本文件(项目说明)
├── README.md                        # 用户视角
├── docs/
│   └── design.md                    # 设计决策 + 调研笔记
├── scripts/                         # 主流程脚本
│   ├── start_ds_v4_flash.sh         # DeepSeek-V4-Flash sglang server(teacher)
│   ├── start_qwen35_9b.sh           # Qwen3.5-9B sglang server(rollout)
│   ├── run_slime_opd_qwen3_8b.sh    # slime OPD: Qwen3-8B student + DS-V4 teacher
│   ├── run_slime_opd_qwen35_9b.sh   # slime OPD: Qwen3.5-9B student + DS-V4 teacher
│   ├── run_slime_grpo_qwen3_8b.sh   # slime GRPO baseline(Qwen3-8B)
│   ├── convert_hf_to_mcore.sh       # 调 slime 官方 convert_hf_to_torch_dist.py
│   ├── smoke_test_imports.sh        # §10.1 import 验证
│   ├── smoke_test_server_qwen35.sh  # §10.2 Qwen3.5-9B sglang server smoke
│   └── eval/
│       └── eval_harness.sh          # 评测入口
├── configs/                         # (slime 用脚本驱动,这里放 launch 配置)
│   ├── slime/
│   ├── verl/
│   └── easyr1/
├── src/                             # ★ 改这里实现新算法
│   ├── slime_plugin/                # slime 自定义 plugin
│   │   ├── evidence_ratio.py        # RLSD 思路(从 RLSD 论文搬)
│   │   └── privileged_teacher.py    # OPSD 思路
│   └── rewards/                     # reward verifier
│       ├── math_verifier.py         # 数学题 reward
│       ├── code_verifier.py         # 代码题 reward
│       └── tool_call_verifier.py    # tool-call reward
├── output/                          # 训练输出 / checkpoint
└── logs/
```

## 复用流程(3 步上手新算法)

1. **新 idea** → 写 50–200 行 `src/slime_plugin/<algo>.py`(slime 接口)
2. **新 reward** → 50 行 `src/rewards/<domain>_verifier.py`
3. **新实验** → 复制 `scripts/run_slime_<template>.sh` 改 5 行

## 关键命令(进入项目)

```bash
source $LZY_ROOT/activate_lzy.sh
opd   # alias:conda activate agentic-opd-train + 自动设置 CUDA_VISIBLE_DEVICES

# 看 env 是否装好 + GPU 可见
bash $LZY_ROOT/projects/agentic-opd/scripts/smoke_test_imports.sh
opd-gpu   # 验 CUDA_VISIBLE_DEVICES 工作

# SII skill baseline check
$LZY_ROOT/envs/agentic-opd-train/bin/python \
    $LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/scripts/check_gpu_baseline.py \
    --env-name agentic-opd-train --phase gpu

# 跑 smoke test
opd-smoke-slime   # slime OPD self-distill,~15-30 min
opd-verl
opd-smoke-verl    # verl GRPO 全参,~30-60 min

# 跑完整训练
bash $LZY_ROOT/projects/agentic-opd/scripts/run_slime_opd_qwen3_8b.sh
```

## 容器化 GPU 关键设置(必读!)

**LZY HPC GPU 实例是容器化的**:`nvidia-container-runtime` 启动时只 expose 部分 `/dev/nvidia*`(可能是 0,1,2,3 也可能是 2,4,6,7,每次启动可能不同)。

`opd` / `opd-verl` alias 会**自动检测**容器内 expose 的 GPU 设备并设 `CUDA_VISIBLE_DEVICES`,把容器 index 重映射成 0..N-1,torch/sglang/vllm 全能识别。

**如果硬跑**:错误会表现为 `RuntimeError: Found no NVIDIA driver on your system` / `CUDA error: invalid device`,但 nvidia-smi 实际能看到卡。

详见 `$LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/references/containerized_gpu.md`。

## 模型

| 用途 | 模型 | 路径 |
|------|------|------|
| text 训练 baseline | Qwen3-8B | `$LZY_ROOT/models/Qwen3-8B` |
| multimodal 训练主 student | Qwen3.5-9B | `$LZY_ROOT/models/Qwen3.5-9B` |
| 旗舰 Dense | qwen3.6-27B | `$LZY_ROOT/models/qwen3.6-27B` |
| 旗舰 MoE | qwen3.6-35B-A3B | `$LZY_ROOT/models/qwen3.6-35B-A3B` |
| teacher | deepseek-v4-flash | `$LZY_ROOT/models/deepseek-v4-flash` |

## 框架选择

- **slime**(主):THUDM 出品,production 验证(GLM-4.5/5/5.1),已原生支持 OPD
- **verl** 0.7.1(备用):Hybrid Engine 成熟
- **EasyR1**(备用):VERL fork for VL
- **OpenRLHF** 0.10.3(备用):Multi-Turn VLM RL

## 关键文档

- **`$LZY_ROOT/docs/agentic-opd-config-v6-cu128.md`** — **当前 v6 规范(CUDA 12.8 baseline,2026-06-12)**
- `$LZY_ROOT/docs/agentic-opd-config.md` — v5 规范(CUDA 12.9,**DEPRECATED**,仅作存档)
- `$LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/SKILL.md` — GPU env 通用 5 层栈(已加 Driver × CUDA 兼容矩阵)
- `$LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/references/use_case_agentic_opd.md` — v5→v6 失败轨迹 + 决策案例
- `$LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/references/containerized_gpu.md` — 容器化 GPU + cu13 spillover 修法
- `$LZY_ROOT/llm-wiki/wiki/research/RLSD.md` — RLSD 论文笔记
- `$LZY_ROOT/llm-wiki/raw/00_InBox/Self-Distilled Reasoner_...md` — OPSD 论文

## 已知限制

- **CUDA 版本:cu128 baseline**(LZY HPC driver 主流 570.x 限制)
  - sglang 0.5.12.post1 必须**从源码编**(无 cu128 wheel)
  - sglang-kernel 改名 `sgl-kernel`,用 0.3.14.post1+cu128 wheel
  - sgl-deep-gemm **cu128 无 wheel**,BF16 满血版不依赖;跑 FP8 大模型需自编
- 当前 sglang 0.5.12.post1 可能不识别 `deepseek_v4` model_type → 临时改 `config.json` 中 `model_type: "deepseek_v4" → "deepseek_v3"`(slime 官方 kimik2 同款 trick)
- Qwen3.5/3.6 GDN 后端需 FlashQLA(已装,需 SM_90+,H200 OK)
- 8 卡跑 235B MoE 显存紧,需 ZeRO-3 + CPU offload

## 待办

- [ ] GPU 实例 smoke test(必须)
- [ ] 写 Qwen3.5-9B 的 `scripts/models/qwen3-9b.sh`(参考官方 qwen3-8B.sh)
- [ ] 写 qwen3.6-35B-A3B 的 `scripts/models/qwen3.6-35b-a3b.sh`(参考官方 qwen3-30B-A3B.sh)
- [ ] 跑通 slime OPD baseline(Qwen3-8B student + DeepSeek-V4-Flash teacher)
