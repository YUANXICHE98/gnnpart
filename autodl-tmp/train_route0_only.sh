#!/bin/bash

# Route0训练脚本 - 集成过拟合监控
# 使用多头注意力GNN进行多跳推理训练，实时监控过拟合信号

echo "开始Route0训练 - 集成过拟合监控..."

# 设置参数
GRAPH_DIR="/root/autodl-tmp/subgraphs"
OUTPUT_DIR="/root/autodl-tmp/models/route0_pure_gnn/checkpoints"
HIDDEN_DIM=256
NUM_LAYERS=6
BATCH_SIZE=16
LEARNING_RATE=0.001
EPOCHS=30
DROPOUT=0.2

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "目录结构:"
ls -la /root/autodl-tmp/models/route0_pure_gnn

# 清理之前的训练结果（可选）
echo "是否清理之前的训练结果？(y/N)"
read -t 10 -n 1 cleanup
if [[ $cleanup == "y" || $cleanup == "Y" ]]; then
    echo "清理之前的检查点..."
    rm -f $OUTPUT_DIR/*.pt
    rm -f $OUTPUT_DIR/metrics.json
    rm -rf $OUTPUT_DIR/logs
    rm -f $OUTPUT_DIR/*overfitting*.png
    rm -f $OUTPUT_DIR/*overfitting*.txt
fi

# 运行训练
cd /root/autodl-tmp/models/route0_pure_gnn

python train.py \
    --graph_dir $GRAPH_DIR \
    --output_dir $OUTPUT_DIR \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --epochs $EPOCHS \
    --dropout $DROPOUT \
    --cuda

# 检查训练结果
echo "训练后检查点目录内容:"
ls -la $OUTPUT_DIR

echo "Route0训练完成！"
echo "输出目录: $OUTPUT_DIR"
echo "检查点: $OUTPUT_DIR/"
echo "过拟合分析: $OUTPUT_DIR/final_overfitting_analysis.png"
echo "分析报告: $OUTPUT_DIR/final_overfitting_report.txt"
echo "TensorBoard日志: $OUTPUT_DIR/logs/"

# 显示最终分析报告
if [ -f "$OUTPUT_DIR/final_overfitting_report.txt" ]; then
    echo ""
    echo "=== 最终过拟合分析报告 ==="
    cat "$OUTPUT_DIR/final_overfitting_report.txt"
fi 