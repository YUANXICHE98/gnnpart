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
