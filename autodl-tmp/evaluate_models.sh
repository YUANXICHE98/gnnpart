#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置共享评估参数
CLEAN_DIR="gnn_dataset_clean/subgraphs"
DISTRACTOR_DIR="gnn_dataset_1000/subgraphs/with_distractors"
OUTPUT_DIR="server_results"
SAMPLE_SIZE=1000
WORKERS=16  # 根据服务器CPU核心数调整

echo "开始评估所有模型..."

# 运行评估
python server_route_evaluation.py \
  --clean_dir $CLEAN_DIR \
  --distractor_dir $DISTRACTOR_DIR \
  --output_dir $OUTPUT_DIR \
  --sample_size $SAMPLE_SIZE \
  --workers $WORKERS \
  --route0_model_path models/route0_pure_gnn/checkpoints/best_model.pt \
  --route1_model_path models/route1_gnn_softmask/checkpoints/best_model.pt \
  --route2_model_path models/route2_swarm_search/checkpoints/best_model.pt \
  --route3_model_path models/route3_diff_explainer/checkpoints/best_model.pt \
  --route4_model_path models/route4_multi_chain/checkpoints/best_model.pt

echo "评估完成! 结果保存在 $OUTPUT_DIR 目录中。" 