#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 安装依赖（仅在服务器上执行）
echo "检查并安装必要的依赖..."
if [ -f "/etc/os-release" ]; then
  # 检测Linux系统类型
  . /etc/os-release
  if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then
    echo "在Ubuntu/Debian上安装依赖..."
    sudo apt-get update
    sudo apt-get install -y libssl-dev
  elif [ "$ID" = "centos" ] || [ "$ID" = "rhel" ] || [ "$ID" = "fedora" ]; then
    echo "在CentOS/RHEL/Fedora上安装依赖..."
    sudo yum install -y openssl-devel
  fi
else
  echo "非Linux系统，跳过依赖安装"
fi

# 设置共享训练参数
GRAPH_DIR="gnn_dataset_clean/subgraphs"
BATCH_SIZE=32
EPOCHS=20

echo "开始训练所有模型..."

# 首先确保所有目录存在
echo "创建所有输出目录..."
mkdir -p models/route0_pure_gnn/checkpoints
mkdir -p models/route1_gnn_softmask/checkpoints
mkdir -p models/route2_swarm_search/checkpoints
mkdir -p models/route3_diff_explainer/checkpoints
mkdir -p models/route4_multi_chain/checkpoints

echo "输出目录创建状态："
ls -la models/route0_pure_gnn
ls -la models/route1_gnn_softmask
ls -la models/route2_swarm_search
ls -la models/route3_diff_explainer
ls -la models/route4_multi_chain

# 训练Route 0: Pure GNN
echo "===== 训练Route 0: Pure GNN ====="
python models/route0_pure_gnn/train.py \
  --graph_dir $GRAPH_DIR \
  --output_dir models/route0_pure_gnn/checkpoints \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --cuda

# 训练Route 1: GNN SoftMask
echo "===== 训练Route 1: GNN SoftMask ====="
python models/route1_gnn_softmask/train.py \
  --graph_dir $GRAPH_DIR \
  --output_dir models/route1_gnn_softmask/checkpoints \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --cuda

# 训练Route 2: Swarm Search
echo "===== 训练Route 2: Swarm Search ====="
python models/route2_swarm_search/train.py \
  --graph_dir $GRAPH_DIR \
  --output_dir models/route2_swarm_search/checkpoints \
  --batch_size $BATCH_SIZE \
  --episodes $EPOCHS

# 训练Route 3: Diff Explainer
echo "===== 训练Route 3: Diff Explainer ====="
python models/route3_diff_explainer/train.py \
  --graph_dir $GRAPH_DIR \
  --output_dir models/route3_diff_explainer/checkpoints \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --cuda

# 训练Route 4: Multi Chain
echo "===== 训练Route 4: Multi Chain ====="
python models/route4_multi_chain/train.py \
  --graph_dir $GRAPH_DIR \
  --output_dir models/route4_multi_chain/checkpoints \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --cuda

echo "所有模型训练完成!"

# 检查训练结果
echo "训练后目录状态检查："
echo "Route 0 检查点目录："
ls -la models/route0_pure_gnn/checkpoints
echo "Route 1 检查点目录："
ls -la models/route1_gnn_softmask/checkpoints
echo "Route 2 检查点目录："
ls -la models/route2_swarm_search/checkpoints
echo "Route 3 检查点目录："
ls -la models/route3_diff_explainer/checkpoints
echo "Route 4 检查点目录："
ls -la models/route4_multi_chain/checkpoints 