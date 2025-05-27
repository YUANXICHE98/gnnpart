"""
工具包，包含训练、数据加载和可视化等功能模块

主要模块：
- gnntrain: GNN模型训练器
- gnn_dataset_reader: 数据加载和预处理
- visualisation: 训练过程和结果可视化
"""

# 清除错误导入
# from .abelian import AbelianGNN          # 错误：utils中没有abelian模块
# from .non_abelian import NonAbelianGNN    # 错误：utils中没有non_abelian模块
# from .identity import IdentityGNN         # 错误：utils中没有identity模块

# 如果需要从utils中导出这些类，应该这样做：
# from models.gnn.abelian import AbelianGNN
# from models.gnn.non_abelian import NonAbelianGNN
# from models.gnn.identity import IdentityGNN

# 修复错误导入方式 - 直接从graph_utils导入函数
from .graph_utils import load_graph_data

# 从visualization模块导入
from .visualization import TrainingVisualizer, ProbabilityVisualizer

# 删除循环导入
# from trainer.gnntrain import GNNTrainer

# # 导入工具函数
# from .group_extension import (
#     extend_abelian_graph,
#     extend_non_abelian_graph,
#     extend_identity_graph
# )

__all__ = [
    # 'GNNTrainer',  # 删除此项以避免循环导入
    'load_graph_data',
    'TrainingVisualizer',
    'ProbabilityVisualizer'
] 