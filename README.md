# Agentic On-Policy Distillation — Infra 项目

> 状态:**CPU 端已装好,等 GPU 端 smoke test**
> 创建:2026-06-01
> 详见:`docs/agentic-opd-config.md`(12 节 + 实施记录)
> 父 wiki:`$LZY_ROOT/llm-wiki/wiki/research/RLSD.md`、`Self-Distilled Reasoner_...md`

## 快速上手(在 GPU 实例)

```bash
# 1. source LZY shell
source $LZY_ROOT/activate_lzy.sh

# 2. 激活 conda env
opd   # 等价于 conda activate agentic-opd-train

# 3. CPU 端 import smoke(确认 env 装好)
opd-smoke   # 19/19 PASS

# 4. GPU 端 sglang server smoke
opd-server-smoke   # 启 Qwen3.5-9B,跑 /v1/models + chat

# 5. 起 DeepSeek-V4-Flash teacher server
opd-flash &   # 后台跑

# 6. 跑 slime OPD(Qwen3-8B student)
opd-opd-qwen3

# 或 Qwen3.5-9B student
opd-opd-qwen35
```

## 文件清单

```
agentic-opd/
├── CLAUDE.md                        # 项目说明
├── README.md                        # 本文件
├── docs/                            # 设计 + 笔记
├── scripts/
│   ├── start_ds_v4_flash.sh         # DeepSeek-V4-Flash teacher
│   ├── start_qwen35_9b.sh           # Qwen3.5-9B rollout
│   ├── smoke_test_imports.sh        # 19 项 import 验证
│   ├── smoke_test_server_qwen35.sh  # GPU 端 server smoke
│   ├── convert_hf_to_mcore.sh       # HF→mcore
│   ├── run_slime_opd_qwen3_8b.sh    # slime OPD (Qwen3-8B)
│   ├── run_slime_opd_qwen35_9b.sh   # slime OPD (Qwen3.5-9B)
│   ├── run_slime_grpo_qwen3_8b.sh   # GRPO baseline
│   └── eval/eval_harness.sh
├── configs/slime/                   # GRPO 启动 yaml
├── src/
│   ├── slime_plugin/                # 改这里写新算法
│   └── rewards/                     # 改这里写新 reward
├── output/  logs/  docs/
```

## 模型(在 `$LZY_ROOT/models/`)

| 用途 | 模型 | 大小 |
|------|------|------|
| text baseline | Qwen3-8B | 16G |
| multimodal student | Qwen3.5-9B | 19G |
| 旗舰 Dense | qwen3.6-27B | 52G |
| 旗舰 MoE | qwen3.6-35B-A3B | 67G |
| teacher | deepseek-v4-flash | 149G |

## 框架

- **slime**(主,THUDM 出品):production 验证(GLM-5.1,Qwen3.5/3.6/3MoE/DeepSeek V3)
- **verl 0.7.1**(备用):Hybrid Engine
- **EasyR1 / OpenRLHF**:暂未装

## 框架支持的关键能力

- `slime/examples/on_policy_distillation/`:**你 wiki OPSD/RLSD 主线对应**
- `slime/examples/coding_agent_rl/`:**完整 SWE agent RL**
- `slime/examples/multi_agent/`, `search-r1/`, `tau-bench/`, `retool/`, `strands_sglang/`, `fully_async/`
- `slime/scripts/models/`:Qwen3 / Qwen3.5 / Qwen3-MoE / DeepSeek-V3 模板全部有

## 实施状态

- CPU 端:**完成**(19/19 import 通过)
- GPU 端:**待跑 smoke**(详见 `docs/agentic-opd-config.md` §13.4)

## 关键 patch(必须在 env 内才能 import)

`kernels/layer/layer.py` 和 `kernels/layer/func.py`:
```python
# 旧:raise ValueError("Either a revision or a version must be specified.")
# 新:revision = revision or "main"  # PATCHED
```
原因:transformers 5.6 升级后调用 `LayerRepository(...)` 不传 revision/version,kernels 0.15 必填。已自动 patch。

## 待办(论文方向)

- [ ] 写 RLSD 思路的 slime plugin(`src/slime_plugin/evidence_ratio.py` 已有占位,需实装)
- [ ] 写多轮 agentic 的 reward verifier(`src/rewards/`)
- [ ] 在 Qwen3-VL-8B-Instruct(注:你 models/ 里没这模型,需下载)上跑 RLSD baseline
- [ ] 写 OPSD/RLSD 论文的方法部分,引用 slime OPD example

## 相关研究

- RLSD 论文:[[../llm-wiki/wiki/sources/Self-Distilled RLVR]] arXiv:2604.03128
- OPSD 论文:[[../llm-wiki/raw/00_InBox/Self-Distilled Reasoner_...md]] arXiv:2601.18734
- SSD 论文:[[../llm-wiki/raw/00_InBox/Embarrassingly Simple Self-Distillation Improves Code Generation]] arXiv:2604.01193
- DeepSeek-V4 paper:`$LZY_ROOT/models/deepseek-v4-flash/DeepSeek_V4.pdf`
