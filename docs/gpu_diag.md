# GPU 实例诊断步骤 — 修 SII skill 报的两个 FAIL

> 跑通后,这两个 FAIL 应该是用户(管理员)修的,不是我(从 CPU 实例)能远程修的
> 因为 L5 要 root 创 /dev/nvidia* 节点

---

## 0. 跑诊断命令(用户在 GPU 实例的 SSH 里)

```bash
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
source $LZY_ROOT/activate_lzy.sh

# --- 1. 验证 NFS 上 env 的 Python 能直接 import torch ---
$LZY_ROOT/envs/agentic-opd-train/bin/python -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cuda.is_available:', torch.cuda.is_available())
print('cuda.device_count:', torch.cuda.device_count())
" 2>&1 | head -20
```

**期望 3 种结果之一**:

| 结果 | 诊断 | 修法 |
|------|------|------|
| ✅ `torch: 2.11.0+cu128 ... cuda.device_count: 8` | torch 没问题,L2 check 是误报 | 改用 env's python 重跑 baseline check |
| ⏱️ 命令卡住不返回 | import torch 在等 /dev/nvidia* 或某 lib | 跑 `ls /dev/nvidia*` 确认;若空要 nvidia-modprobe |
| ❌ 报 ImportError 或 libxxx.so not found | GPU 实例 base image 缺某个 lib | 缺啥装啥,常见: `libnuma`, `libstdc++` 等 |

```bash
# --- 2. 验证 L5: /dev/nvidia* 设备节点 ---
ls -la /dev/nvidia* 2>&1
```

**期望**:看到 `nvidia0`, `nvidia-uvm`, `nvidiactl`(8 卡机器有 nvidia0..nvidia7)

**若空**:
```bash
sudo nvidia-modprobe    # 创 device nodes
# 或(若 nvidia-modprobe 不可用)
sudo modprobe nvidia
sudo modprobe nvidia-uvm
sudo modprobe nvidia-drm    # 可选
```

```bash
# --- 3. 验证 libcuda.so.1 能 dlopen ---
$LZY_ROOT/envs/agentic-opd-train/bin/python -c "
import ctypes
try:
    ctypes.CDLL('libcuda.so.1')
    print('libcuda.so.1 OK')
except OSError as e:
    print('FAIL:', e)
" 2>&1
```

**若 FAIL**:
```bash
# 找 libcuda.so.1 在哪
find / -name "libcuda.so.1" 2>/dev/null | head -5
# 通常在 /lib/x86_64-linux-gnu/ 或 /usr/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

```bash
# --- 4. 验证 base image 的 Python 工具链 ---
ldd $LZY_ROOT/envs/agentic-opd-train/bin/python3.12 2>&1 | head -20
# 期望:所有 .so 都 resolved
# 若有 "not found":GPU base image 缺某个基础 lib
```

```bash
# --- 5. 完整重跑 baseline check(用 env's Python) ---
$LZY_ROOT/envs/agentic-opd-train/bin/python \
    $LZY_ROOT/projects/shared-agent-skills/SII_GPU_env_construct/scripts/check_gpu_baseline.py \
    --env-name agentic-opd-train --phase gpu
```

---

## 1. 修了 L5 之后还会看到的:env 的 cu13 spillover

slime env 的 `nvidia sub-packages` 列表里有 `cu13` 和 `nvshmem` —— 这是 sglang 0.5.12.post1 装时"溢出"的 cu13 库。

v4 文档 §5.3.3 写了清理步骤,可能当时没全跑通。若 GPU 上 torch import 跟 cu13 冲突,可以再跑一次清理:

```bash
opd   # 激活 slime env

# 卸 cu13 残留
pip uninstall -y \
    nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
    nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 \
    nvidia-nvtx nvidia-cutlass-dsl-libs-cu13 2>/dev/null || true

# 重装 cu12 等价
pip install --force-reinstall --no-deps \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
```

**注意**:这步会替换一些 CUDA 库,**先确认 smoke test 跑通再做**。verl env 干净,不用动。

---

## 2. 修完后该看到的

```
[✓] L1 project code: OK
[✓] L2 conda env: OK
        numpy: 1.26.4
        torch: 2.6.0+cu126
[✓] L3 system shared libs: OK
[✓] L4 CUDA backend: torch.cuda.is_available() True, 8 GPU
[✓] L5 kernel module: /dev/nvidia0..7 present
Overall: PASS
```

然后跑:
```bash
opd-smoke-slime   # 验 slime OPD 管线
opd-smoke-verl    # 验 verl GRPO 管线
```

详细见 `$LZY_ROOT/projects/agentic-opd/docs/smoke_tests.md`。
