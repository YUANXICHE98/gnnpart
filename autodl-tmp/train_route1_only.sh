#!/bin/bash

echo "===== Route 1 GNN SoftMask 训练脚本 (改进版) ====="

# 检查OpenSSL 3.0是否已安装
if [ ! -f "/opt/openssl3/bin/openssl" ]; then
    echo "未检测到OpenSSL 3.0安装，尝试安装..."
    if [ -f "./fix.sh" ]; then
        bash ./fix.sh
    else
        echo "错误: 找不到OpenSSL 3.0安装脚本!"
        exit 1
    fi
fi

# 设置环境变量
echo "设置OpenSSL 3.0环境变量..."
export PATH=/opt/openssl3/bin:$PATH
export LD_LIBRARY_PATH=/opt/openssl3/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/openssl3/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=/opt/openssl3/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/openssl3/include:$CPLUS_INCLUDE_PATH
export OPENSSL_DIR=/opt/openssl3
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 验证OpenSSL版本
echo "验证OpenSSL版本..."
/opt/openssl3/bin/openssl version

# 检查库文件版本
echo "检查库文件..."
ls -la /opt/openssl3/lib/libssl.so.3
ls -la /opt/openssl3/lib/libcrypto.so.3
strings /opt/openssl3/lib/libcrypto.so.3 | grep -i "OPENSSL_3.0"

# 设置训练参数 - 根据改进的自适应稀疏控制器调整
GRAPH_DIR="gnn_dataset_clean/subgraphs"
BATCH_SIZE=16  # 减小批大小以提高稳定性
EPOCHS=30      # 增加训练轮数以充分利用自适应机制
LEARNING_RATE=0.0005  # 降低学习率以提高稳定性
SPARSITY_TARGET=0.25  # 明确设置目标稀疏度

# 确保输出目录存在
echo "创建输出目录..."
mkdir -p models/route1_gnn_softmask/checkpoints

echo "目录结构:"
ls -la models/route1_gnn_softmask

# 清理之前的训练结果（可选）
echo "是否清理之前的训练结果？(y/N)"
read -t 10 -n 1 cleanup
if [[ $cleanup == "y" || $cleanup == "Y" ]]; then
    echo "清理之前的检查点..."
    rm -f models/route1_gnn_softmask/checkpoints/*.pt
    rm -f models/route1_gnn_softmask/checkpoints/metrics.json
    rm -rf models/route1_gnn_softmask/checkpoints/logs
fi

# 使用特殊的LD_PRELOAD设置运行训练脚本
echo "开始训练Route 1模型 (改进的自适应稀疏控制)..."
echo "训练参数:"
echo "  - 批大小: $BATCH_SIZE"
echo "  - 训练轮数: $EPOCHS"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 目标稀疏度: $SPARSITY_TARGET"
echo "  - 预训练轮数: 3"
echo "  - 自适应权重范围: 0.001-0.05"

export LD_PRELOAD=/opt/openssl3/lib/libcrypto.so.3:/opt/openssl3/lib/libssl.so.3

python models/route1_gnn_softmask/train.py \
  --graph_dir $GRAPH_DIR \
  --output_dir models/route1_gnn_softmask/checkpoints \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LEARNING_RATE \
  --sparsity_target $SPARSITY_TARGET \
  --hidden_dim 256 \
  --num_layers 2 \
  --dropout 0.2 \
  --cuda

# 检查训练结果
echo "训练后检查点目录内容:"
ls -la models/route1_gnn_softmask/checkpoints

echo "Route 1模型训练完成!" 