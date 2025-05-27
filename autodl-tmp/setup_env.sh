#!/bin/bash
# setup_env.sh - 创建CUDA支持的GNN环境 (使用pip安装加速)

echo "===== 开始配置GNN环境 (CUDA版本) ====="

# 创建新环境
echo "创建新的conda环境..."
conda create -n gnn_env python=3.8 -y

# 激活环境
echo "激活环境..."
source activate gnn_env

# 使用pip直接安装PyTorch CUDA版本
echo "使用pip安装PyTorch (CUDA 11.3版本)..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装其他核心依赖
echo "安装核心依赖..."
pip install pandas matplotlib seaborn scikit-learn tqdm

# 安装DGL
echo "安装DGL (CUDA版本)..."
pip install dgl-cu113==0.9.1

# 安装其他必要依赖
echo "安装其他依赖..."
pip install tensorboardX tabulate networkx

# 验证安装
echo "验证安装..."
# 创建一个简单的测试脚本验证安装
cat > test_imports.py << EOL
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import networkx as nx

try:
    import dgl
    dgl_imported = True
except ImportError as e:
    dgl_imported = False
    dgl_error = str(e)

try:
    from tensorboardX import SummaryWriter
    tb_imported = True
except ImportError:
    tb_imported = False

print("\n===== 环境验证 =====")
print(f"PyTorch版本: {torch.__version__}")
print(f"Pandas版本: {pd.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

if dgl_imported:
    print(f"DGL版本: {dgl.__version__}")
else:
    print(f"DGL导入失败: {dgl_error}")

print(f"TensorboardX可用: {tb_imported}")
print("测试完成!")
EOL

# 运行测试
python test_imports.py

# 设置训练和评估脚本
echo "添加环境激活到训练和评估脚本..."

# 添加激活环境到训练脚本
if [ -f "train_all_models.sh" ]; then
    cp train_all_models.sh train_all_models.sh.bak
    cat > train_all_models_new.sh << EOL
#!/bin/bash
# 激活conda环境
source activate gnn_env

$(cat train_all_models.sh)
EOL
    mv train_all_models_new.sh train_all_models.sh
    chmod +x train_all_models.sh
    echo "已更新train_all_models.sh添加环境激活"
fi

# 添加激活环境到评估脚本
if [ -f "run_and_visualize.sh" ]; then
    cp run_and_visualize.sh run_and_visualize.sh.bak
    cat > run_and_visualize_new.sh << EOL
#!/bin/bash
# 激活conda环境
source activate gnn_env

$(cat run_and_visualize.sh)
EOL
    mv run_and_visualize_new.sh run_and_visualize.sh
    chmod +x run_and_visualize.sh
    echo "已更新run_and_visualize.sh添加环境激活"
fi

echo "===== 环境配置完成 ====="
echo "使用方法: source activate gnn_env"
echo "现在可以运行: ./train_all_models.sh"
echo "然后运行: ./run_and_visualize.sh"