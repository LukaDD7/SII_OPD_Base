#!/bin/bash
# ==============================================================================
# smoke_test_imports.sh — CPU 实例 import 验证
# ==============================================================================
# 验证 env 装好,所有关键 import 通过
# 注:CUDA 算子相关的 import 在 CPU 上会 fail,这是正常的;只验证 Python 层
#
# 用法:
#   bash smoke_test_imports.sh
# ==============================================================================

set -e
export LZY_ROOT=/inspire/hdd/global_user/mengweicheng-240108120092/lzy
source $LZY_ROOT/miniconda3/etc/profile.d/conda.sh
conda activate agentic-opd-train

PASS=0
FAIL=0

check() {
    local name=$1
    local cmd=$2
    if eval "$cmd" > /dev/null 2>&1; then
        echo "[OK]   $name"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $name"
        echo "       cmd: $cmd"
        FAIL=$((FAIL+1))
    fi
}

version() {
    local name=$1
    local cmd=$2
    local v=$(eval "$cmd" 2>/dev/null)
    if [ -n "$v" ]; then
        echo "[OK]   $name = $v"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $name (no version)"
        FAIL=$((FAIL+1))
    fi
}

echo "=========================================="
echo "agentic-opd-train import smoke test"
echo "=========================================="
echo "Python: $(which python)"
echo "Env: $CONDA_PREFIX"
echo ""

echo "--- Compilers & libs ---"
check "gcc" "which gcc"
check "g++" "which g++"
check "gfortran" "which gfortran"
check "rustc" "which rustc"
check "libstdc++ GLIBCXX_3.4.30" "strings \$CONDA_PREFIX/lib/libstdc++.so.6 | grep -q GLIBCXX_3.4.30"
check "libnuma" "python -c 'import ctypes; ctypes.CDLL(\"libnuma.so.1\")'"

echo ""
echo "--- Python packages ---"
version "torch" "python -c 'import torch; print(torch.__version__)'"
version "transformers" "python -c 'import transformers; print(transformers.__version__)'"
version "sglang" "python -c 'import sglang; print(sglang.__version__)' 2>&1 | tail -1"
check "slime (git clone)" "test -d $LZY_ROOT/repos/slime"
check "Megatron-LM (git clone)" "test -d $LZY_ROOT/repos/Megatron-LM"
check "sglang source" "test -d $LZY_ROOT/repos/sglang"
version "peft" "python -c 'import peft; print(peft.__version__)'"
version "trl" "python -c 'import trl; print(trl.__version__)'"

echo ""
echo "--- Models in $LZY_ROOT/models ---"
for m in Qwen3-8B Qwen3.5-9B qwen3.6-27B qwen3.6-35B-A3B deepseek-v4-flash; do
    if [ -d $LZY_ROOT/models/$m ]; then
        size=$(du -sh $LZY_ROOT/models/$m 2>/dev/null | cut -f1)
        echo "[OK]   $m ($size)"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $m missing"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "--- Model config arch ---"
python -c "
import json
for m in ['Qwen3-8B', 'Qwen3.5-9B', 'qwen3.6-27B', 'qwen3.6-35B-A3B', 'deepseek-v4-flash']:
    p = f'$LZY_ROOT/models/{m}/config.json'
    try:
        c = json.load(open(p))
        print(f'  {m}: type={c[\"model_type\"]}, arch={c[\"architectures\"]}')
    except Exception as e:
        print(f'  {m}: ERROR {e}')
" 2>&1

echo ""
echo "=========================================="
echo "PASS: $PASS   FAIL: $FAIL"
echo "=========================================="
if [ $FAIL -gt 0 ]; then
    echo ""
    echo "失败项需排查;GPU 端 smoke test 见 docs/agentic-opd-config.md §10"
    exit 1
fi
echo ""
echo "CPU 端 import 全部通过。下一步:在 GPU 实例跑 §10 完整 smoke test"
