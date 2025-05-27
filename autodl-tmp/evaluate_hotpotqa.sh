#!/bin/bash

# 设置环境变量
# 设置工作目录变量
export WORKDIR=~/autodl-tmp

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$WORKDIR


# 设置共享评估参数
CLEAN_DIR="gnn_dataset_clean/subgraphs"
HOTPOTQA_FILE="hotpotqa_dataset/hotpot_dev_fullwiki.json"
OUTPUT_DIR="hotpotqa_results"
SAMPLE_SIZE=500
WORKERS=16  # 根据服务器CPU核心数调整

echo "开始在HotpotQA数据集上评估模型..."

# 运行HotpotQA评估
python server_route_evaluation.py \
  --clean_dir $CLEAN_DIR \
  --hotpotqa_file $HOTPOTQA_FILE \
  --hotpotqa_output $OUTPUT_DIR \
  --sample_size $SAMPLE_SIZE \
  --workers $WORKERS \
  --only_hotpotqa \
  --route0_model_path models/route0_pure_gnn/checkpoints/best_model.pt \
  --route1_model_path models/route1_gnn_softmask/checkpoints/best_model.pt \
  --route2_model_path models/route2_swarm_search/checkpoints/best_model.pt \
  --route3_model_path models/route3_diff_explainer/checkpoints/best_model.pt \
  --route4_model_path models/route4_multi_chain/checkpoints/best_model.pt

echo "HotpotQA评估完成! 结果保存在 $OUTPUT_DIR 目录中。" 