#!/bin/bash
# 激活conda环境
source activate gnn_env

#!/bin/bash
# 激活conda环境
source activate gnn_env

#!/bin/bash
# 激活conda环境
source activate gnn_env

#!/bin/bash
# run_and_visualize.sh
# 运行模型评估并自动生成可视化结果

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 设置共享评估参数
CLEAN_DIR="gnn_dataset_clean/subgraphs"
DISTRACTOR_DIR="gnn_dataset_1000/subgraphs/with_distractors"
OUTPUT_DIR="server_results"
SAMPLE_SIZE=1000
WORKERS=16  # 根据服务器CPU核心数调整

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== 开始评估所有模型... ===${NC}"

# 1. 运行评估
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

echo -e "${GREEN}✓ 评估完成! 结果保存在 $OUTPUT_DIR 目录中${NC}"

# 2. 自动生成可视化结果
echo -e "${YELLOW}=== 开始生成可视化结果... ===${NC}"

# 创建可视化目录
VIZ_DIR="${OUTPUT_DIR}/visualization"
mkdir -p $VIZ_DIR

# 使用Python生成可视化
python -c "
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from pathlib import Path
import seaborn as sns

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print('警告: 未能设置中文字体，可能影响中文显示效果')

# 设置风格
plt.style.use('ggplot')
sns.set(style='whitegrid')

# 读取结果数据
results_dir = '${OUTPUT_DIR}'
results_file = os.path.join(results_dir, 'route_comparison.json')
hotpotqa_file = os.path.join(results_dir, 'hotpotqa_results.json')

# 检查文件是否存在
if not os.path.exists(results_file):
    print(f'警告: 找不到主要结果文件 {results_file}')
    has_results = False
else:
    with open(results_file, 'r') as f:
        results = json.load(f)
    has_results = True

has_hotpotqa = os.path.exists(hotpotqa_file)
if has_hotpotqa:
    with open(hotpotqa_file, 'r') as f:
        hotpotqa_results = json.load(f)

# 可视化输出目录
viz_dir = '${VIZ_DIR}'
os.makedirs(viz_dir, exist_ok=True)

# ===== 1. 模型性能对比图 =====
if has_results:
    model_names = ['纯GNN (Route 0)', '软掩码GNN (Route 1)', '群搜索GNN (Route 2)', 
                  '差分解释器 (Route 3)', '多链GNN (Route 4)']
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    # 清洁数据集结果
    clean_values = []
    for i in range(5):  # 5条路由
        route_key = f'route{i}'
        if route_key in results.get('clean', {}):
            values = [results['clean'][route_key].get(metric, 0) for metric in metrics]
            clean_values.append(values)
        else:
            clean_values.append([0, 0, 0, 0])
    
    # 干扰节点数据集结果
    distractor_values = []
    for i in range(5):  # 5条路由
        route_key = f'route{i}'
        if route_key in results.get('distractor', {}):
            values = [results['distractor'][route_key].get(metric, 0) for metric in metrics]
            distractor_values.append(values)
        else:
            distractor_values.append([0, 0, 0, 0])
    
    # 创建性能对比图
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 清洁数据集子图
    clean_df = pd.DataFrame(clean_values, columns=metric_names, index=model_names)
    clean_df.plot(kind='bar', ax=axes[0], rot=30, colormap='viridis')
    axes[0].set_title('清洁数据集性能', fontsize=16)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('分数', fontsize=14)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 添加数值标签
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.2f', fontsize=10)
    
    # 干扰节点数据集子图
    distractor_df = pd.DataFrame(distractor_values, columns=metric_names, index=model_names)
    distractor_df.plot(kind='bar', ax=axes[1], rot=30, colormap='viridis')
    axes[1].set_title('干扰节点数据集性能', fontsize=16)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel('分数', fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # 添加数值标签
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_performance.png'), dpi=300, bbox_inches='tight')
    print(f'生成模型性能对比图: {os.path.join(viz_dir, 'model_performance.png')}')
    
    # ===== 2. 鲁棒性对比图 =====
    # 计算清洁数据集和干扰数据集之间的性能差异
    robustness_data = []
    for i in range(5):
        route_key = f'route{i}'
        clean_acc = results.get('clean', {}).get(route_key, {}).get('accuracy', 0)
        distractor_acc = results.get('distractor', {}).get(route_key, {}).get('accuracy', 0)
        robustness = distractor_acc / clean_acc if clean_acc > 0 else 0
        robustness_data.append(robustness)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_names, robustness_data, color='skyblue')
    plt.title('模型鲁棒性对比 (干扰节点/清洁数据集准确率比值)', fontsize=16)
    plt.xlabel('模型', fontsize=14)
    plt.ylabel('鲁棒性比值', fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_robustness.png'), dpi=300, bbox_inches='tight')
    print(f'生成模型鲁棒性对比图: {os.path.join(viz_dir, 'model_robustness.png')}')
    
    # ===== 3. 处理时间对比图 =====
    time_data = []
    for i in range(5):
        route_key = f'route{i}'
        clean_time = results.get('clean', {}).get(route_key, {}).get('processing_time', 0)
        distractor_time = results.get('distractor', {}).get(route_key, {}).get('processing_time', 0)
        # 取平均
        avg_time = (clean_time + distractor_time) / 2
        time_data.append(avg_time)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_names, time_data, color='salmon')
    plt.title('模型处理时间对比 (秒/样本)', fontsize=16)
    plt.xlabel('模型', fontsize=14)
    plt.ylabel('平均处理时间 (秒)', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}s', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_processing_time.png'), dpi=300, bbox_inches='tight')
    print(f'生成模型处理时间对比图: {os.path.join(viz_dir, 'model_processing_time.png')}')

# ===== 4. HotpotQA评估结果可视化 =====
if has_hotpotqa:
    # 提取HotpotQA结果
    qa_metrics = {}
    for route_key, route_data in hotpotqa_results.items():
        if 'metrics' in route_data:
            qa_metrics[route_key] = {
                'em': route_data['metrics'].get('exact_match', 0),
                'f1': route_data['metrics'].get('f1', 0),
                'precision': route_data['metrics'].get('precision', 0),
                'recall': route_data['metrics'].get('recall', 0)
            }
    
    if qa_metrics:
        model_names = []
        em_scores = []
        f1_scores = []
        
        for route_key in sorted(qa_metrics.keys()):
            route_num = route_key.replace('route', '')
            model_name = f'路由 {route_num}'
            model_names.append(model_name)
            em_scores.append(qa_metrics[route_key]['em'])
            f1_scores.append(qa_metrics[route_key]['f1'])
        
        # 创建分组柱状图
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, em_scores, width, label='精确匹配')
        rects2 = ax.bar(x + width/2, f1_scores, width, label='F1分数')
        
        ax.set_ylabel('分数', fontsize=14)
        ax.set_title('HotpotQA评估结果', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords='offset points',
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'hotpotqa_results.png'), dpi=300, bbox_inches='tight')
        print(f'生成HotpotQA评估结果图: {os.path.join(viz_dir, 'hotpotqa_results.png')}')

# ===== 5. 生成HTML报告 =====
html_output = os.path.join(viz_dir, 'evaluation_report.html')

html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN模型评估报告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2980b9;
            margin-top: 30px;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .highlight {{
            background-color: #e3f2fd;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            margin-top: 50px;
        }}
    </style>
</head>
<body>
    <h1>GNN模型评估报告</h1>
    
    <h2>1. 模型性能对比</h2>
    <div class="image-container">
        <img src="model_performance.png" alt="模型性能对比">
    </div>
    
    <h2>2. 模型鲁棒性分析</h2>
    <div class="image-container">
        <img src="model_robustness.png" alt="模型鲁棒性对比">
    </div>
    
    <h2>3. 模型处理时间</h2>
    <div class="image-container">
        <img src="model_processing_time.png" alt="模型处理时间对比">
    </div>
'''

if has_hotpotqa:
    html_content += '''
    <h2>4. HotpotQA评估结果</h2>
    <div class="image-container">
        <img src="hotpotqa_results.png" alt="HotpotQA评估结果">
    </div>
    '''

# 添加汇总表格
if has_results:
    best_model_idx = np.argmax([results.get('clean', {}).get(f'route{i}', {}).get('accuracy', 0) for i in range(5)])
    best_robust_idx = np.argmax(robustness_data)
    best_time_idx = np.argmin(time_data)
    
    html_content += f'''
    <h2>5. 评估结果汇总</h2>
    <table>
        <tr>
            <th>模型</th>
            <th>清洁数据集准确率</th>
            <th>干扰节点准确率</th>
            <th>鲁棒性比值</th>
            <th>处理时间 (秒)</th>
        </tr>
    '''
    
    for i in range(5):
        route_key = f'route{i}'
        clean_acc = results.get('clean', {}).get(route_key, {}).get('accuracy', 0)
        distractor_acc = results.get('distractor', {}).get(route_key, {}).get('accuracy', 0)
        robustness = robustness_data[i]
        proc_time = time_data[i]
        
        highlight = ''
        if i == best_model_idx:
            highlight = ' class="highlight"'
        
        html_content += f'''
        <tr{highlight}>
            <td>{model_names[i]}</td>
            <td>{clean_acc:.4f}</td>
            <td>{distractor_acc:.4f}</td>
            <td>{robustness:.4f}</td>
            <td>{proc_time:.4f}</td>
        </tr>
        '''
    
    html_content += '''
    </table>
    '''

# 添加结论部分
if has_results:
    best_model = model_names[best_model_idx]
    best_robust = model_names[best_robust_idx]
    best_time = model_names[best_time_idx]
    
    html_content += f'''
    <h2>6. 结论</h2>
    <ul>
        <li><strong>最佳性能模型:</strong> {best_model}</li>
        <li><strong>最佳鲁棒性模型:</strong> {best_robust}</li>
        <li><strong>最快处理速度模型:</strong> {best_time}</li>
    </ul>
    '''

# 添加时间戳
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
html_content += f'''
    <p class="timestamp">报告生成时间: {timestamp}</p>
</body>
</html>
'''

with open(html_output, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'生成HTML评估报告: {html_output}')
"

echo -e "${GREEN}✓ 可视化结果生成完成！结果保存在 ${VIZ_DIR} 目录中${NC}"

# 生成评估摘要文本报告
echo -e "${YELLOW}=== 生成评估摘要报告... ===${NC}"

python -c "
import os
import json
import numpy as np
from tabulate import tabulate
from datetime import datetime

# 读取结果数据
results_dir = '${OUTPUT_DIR}'
results_file = os.path.join(results_dir, 'route_comparison.json')

if not os.path.exists(results_file):
    print('警告: 找不到主要结果文件')
    exit(1)

with open(results_file, 'r') as f:
    results = json.load(f)

# 准备表格数据
model_names = ['纯GNN (Route 0)', '软掩码GNN (Route 1)', '群搜索GNN (Route 2)', 
              '差分解释器 (Route 3)', '多链GNN (Route 4)']
metrics = ['准确率', '精确率', '召回率', 'F1分数', '处理时间(秒)']

# 清洁数据集结果
clean_data = []
for i in range(5):
    route_key = f'route{i}'
    if route_key in results.get('clean', {}):
        row = [
            model_names[i],
            results['clean'][route_key].get('accuracy', 0),
            results['clean'][route_key].get('precision', 0),
            results['clean'][route_key].get('recall', 0),
            results['clean'][route_key].get('f1', 0),
            results['clean'][route_key].get('processing_time', 0),
        ]
        clean_data.append(row)

# 干扰节点数据集结果
distractor_data = []
for i in range(5):
    route_key = f'route{i}'
    if route_key in results.get('distractor', {}):
        row = [
            model_names[i],
            results['distractor'][route_key].get('accuracy', 0),
            results['distractor'][route_key].get('precision', 0),
            results['distractor'][route_key].get('recall', 0),
            results['distractor'][route_key].get('f1', 0),
            results['distractor'][route_key].get('processing_time', 0),
        ]
        distractor_data.append(row)

# 计算鲁棒性
robustness_data = []
for i in range(5):
    route_key = f'route{i}'
    clean_acc = results.get('clean', {}).get(route_key, {}).get('accuracy', 0)
    distractor_acc = results.get('distractor', {}).get(route_key, {}).get('accuracy', 0)
    robustness = distractor_acc / clean_acc if clean_acc > 0 else 0
    row = [model_names[i], robustness]
    robustness_data.append(row)

# 找出最佳模型
if clean_data:
    best_model_idx = np.argmax([row[1] for row in clean_data])
    best_model = clean_data[best_model_idx][0]
    best_acc = clean_data[best_model_idx][1]
else:
    best_model = '未知'
    best_acc = 0

# 找出最鲁棒模型
if robustness_data:
    best_robust_idx = np.argmax([row[1] for row in robustness_data])
    best_robust = robustness_data[best_robust_idx][0]
    best_robust_score = robustness_data[best_robust_idx][1]
else:
    best_robust = '未知'
    best_robust_score = 0

# 找出最快模型
if clean_data:
    best_time_idx = np.argmin([row[5] for row in clean_data])
    best_time = clean_data[best_time_idx][0]
    best_time_score = clean_data[best_time_idx][5]
else:
    best_time = '未知'
    best_time_score = 0

# 生成摘要报告
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
report = f'''
====================================================================
                        GNN模型评估摘要报告
====================================================================
生成时间: {timestamp}

--------------------------------------------------------------------
                     清洁数据集评估结果
--------------------------------------------------------------------
{tabulate([[row[0], f'{row[1]:.4f}', f'{row[2]:.4f}', f'{row[3]:.4f}', f'{row[4]:.4f}', f'{row[5]:.4f}'] for row in clean_data], 
          headers=['模型', '准确率', '精确率', '召回率', 'F1分数', '处理时间(秒)'], 
          tablefmt='grid')}

--------------------------------------------------------------------
                    干扰节点数据集评估结果
--------------------------------------------------------------------
{tabulate([[row[0], f'{row[1]:.4f}', f'{row[2]:.4f}', f'{row[3]:.4f}', f'{row[4]:.4f}', f'{row[5]:.4f}'] for row in distractor_data], 
          headers=['模型', '准确率', '精确率', '召回率', 'F1分数', '处理时间(秒)'], 
          tablefmt='grid')}

--------------------------------------------------------------------
                        模型鲁棒性分析
--------------------------------------------------------------------
{tabulate([[row[0], f'{row[1]:.4f}'] for row in robustness_data], 
          headers=['模型', '鲁棒性比值 (干扰/清洁)'], 
          tablefmt='grid')}

--------------------------------------------------------------------
                           结论
--------------------------------------------------------------------
* 最佳性能模型: {best_model} (清洁数据集准确率: {best_acc:.4f})
* 最佳鲁棒性模型: {best_robust} (鲁棒性比值: {best_robust_score:.4f})
* 最快处理速度模型: {best_time} (处理时间: {best_time_score:.4f}秒)

====================================================================
      详细可视化结果请查看: ${VIZ_DIR}/evaluation_report.html
====================================================================
'''

# 保存报告
report_file = os.path.join('${OUTPUT_DIR}', 'evaluation_summary.txt')
with open(report_file, 'w') as f:
    f.write(report)

print(f'评估摘要报告已保存到: {report_file}')

# 打印到控制台
print('\\n' + report)
"

echo -e "${GREEN}✓ 评估摘要报告生成完成！${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}所有评估和可视化已完成。请查看以下文件获取结果:${NC}"
echo -e "${BLUE}  - 可视化HTML报告: ${VIZ_DIR}/evaluation_report.html${NC}"
echo -e "${BLUE}  - 评估摘要报告: ${OUTPUT_DIR}/evaluation_summary.txt${NC}"
echo -e "${BLUE}  - 可视化图表目录: ${VIZ_DIR}/${NC}"
echo -e "${BLUE}======================================================${NC}" 
