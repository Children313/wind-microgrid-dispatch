#!/bin/bash
# setup.sh
set -e

echo "[setup] 检测 GPU..."
GPU_FLAG="cpu"
if [[ "$*" == *"--gpu"* ]]; then GPU_FLAG="gpu"
elif [[ "$*" == *"--cpu"* ]]; then GPU_FLAG="cpu"
elif command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then GPU_FLAG="gpu"
fi
echo "[setup] mode: $GPU_FLAG"

echo "[setup] 安装 PyTorch..."
if [ "$GPU_FLAG" = "gpu" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/cu121 || \
    pip install torch
else
    pip install torch --index-url https://download.pytorch.org/whl/cpu || \
    pip install torch
fi

echo "[setup] 装其他依赖..."
pip install -r requirements.txt

# 关键: 修 pymgrid 1.4.1 的 numpy 2.0 兼容性问题
echo "[setup] 修补 pymgrid..."
PYMGRID_PATH=$(python -c "import pymgrid, os; print(os.path.dirname(pymgrid.__file__))")
BASE_MODULE="${PYMGRID_PATH}/modules/base/base_module.py"
if [ -f "$BASE_MODULE" ] && grep -q "np.product(" "$BASE_MODULE"; then
    sed -i.bak 's/np\.product(/np.prod(/g' "$BASE_MODULE"
    echo "  ✓ patched np.product -> np.prod"
fi

echo "[setup] 验证..."
python -c "
import torch, pymgrid, stable_baselines3
print(f'  torch:     {torch.__version__} (cuda={torch.cuda.is_available()})')
print(f'  pymgrid:   {pymgrid.__version__}')
print(f'  sb3:       {stable_baselines3.__version__}')
"
echo "✓ 完成"
