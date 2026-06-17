#!/bin/bash
# ==============================================================================
# check_gpu_visible.sh — 验证 CUDA_VISIBLE_DEVICES 工作的 quick test
# ==============================================================================
# 用法(GPU 实例):
#   source $LZY_ROOT/activate_lzy.sh
#   opd   # 自动检测 + export CUDA_VISIBLE_DEVICES
#   bash $LZY_ROOT/projects/agentic-opd/scripts/check_gpu_visible.sh
#
# 期望输出:
#   CUDA_VISIBLE_DEVICES = 2,4,6,7  (或其他,你这次容器的 index)
#   torch.cuda.is_available: True
#   device_count: 4
#     device 0: NVIDIA H200
#     ...
# ==============================================================================

echo "=========================================="
echo "GPU visible check"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-'(not set)'}"
echo "nvidia-smi visible devices:"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | head -10
echo ""

# 单行 python -c,避免多行缩进问题
$LZY_ROOT/envs/agentic-opd-train/bin/python -c "
import os, torch
print('--- torch view ---')
print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)'))
print('torch.cuda.is_available:', torch.cuda.is_available())
print('torch.cuda.device_count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  device {i}:', torch.cuda.get_device_name(i))
"

echo ""
echo "=========================================="
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[WARN] CUDA_VISIBLE_DEVICES 未设 — torch 看到的 device 跟 nvidia-smi 物理 index 一致"
    echo "       跑 opd alias 会自动设"
else
    echo "[OK] CUDA_VISIBLE_DEVICES 已设,torch 看到 logical 0..N-1"
fi
echo "=========================================="
