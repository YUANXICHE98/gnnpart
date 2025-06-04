# 原始标签数据
labels = [
    "EO", "EI", "ILF", "EO", "EI", "EQ", "EI", "ILF", "ILF", "ILF",
    "EI", "ILF", "ILF", "ILF", "EQ", "EO", "EO", "EO", "EO", "EO",
    "ILF", "ILF", "ILF", "EQ", "ILF", "ILF", "ILF", "EQ", "ILF", "ILF",
    "EQ", "ILF", "ILF", "EQ", "ILF", "EQ", "ILF", "EQ", "ILF", "EQ",
    "ILF", "EQ", "ILF", "EQ", "EQ", "EI", "EO", "EI", "ILF", "EO",
    "ILF", "EO", "ILF", "EO", "ILF", "ILF", "EO", "ILF", "EO", "ILF",
    "EO", "ILF", "EI", "ILF", "EO", "ILF", "EO", "ILF", "EO", "EO",
    "EO", "EQ", "ILF", "EO", "ILF", "EO", "ILF", "EO", "ILF", "EO",
    "EI", "EO", "ILF", "EI", "EO", "EO", "EO", "EO"
]

# 功能点权重配置（标准权重-简单级别）
function_point_weights = {
    "EI": 3,    # 外部输入 (External Input)
    "EO": 4,    # 外部输出 (External Output)  
    "EQ": 3,    # 外部查询 (External Inquiry)
    "ILF": 7,   # 内部逻辑文件 (Internal Logical File)
    "EIF": 5    # 外部接口文件 (External Interface File)
}

# 功能点类型中文名称
function_point_names = {
    "EI": "外部输入",
    "EO": "外部输出",
    "EQ": "外部查询", 
    "ILF": "内部逻辑文件",
    "EIF": "外部接口文件"
}

# 统计标签数量
def count_labels(labels):
    """统计标签出现次数"""
    label_count = {}
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count

# 计算功能点
def calculate_function_points(label_statistics):
    """计算功能点总数"""
    total_fp = 0
    fp_details = {}
    
    for label, count in label_statistics.items():
        if label in function_point_weights:
            weight = function_point_weights[label]
            fp = count * weight
            total_fp += fp
            fp_details[label] = {
                'count': count,
                'weight': weight,
                'function_points': fp,
                'name': function_point_names[label]
            }
    
    return total_fp, fp_details

# 计算统计结果
label_statistics = count_labels(labels)
total_fp, fp_details = calculate_function_points(label_statistics)

# 输出基本统计结果
print("=" * 50)
print("功能点分析报告")
print("=" * 50)

print("\n1. 基本统计结果：")
print("-" * 30)
total_count = len(labels)

for label, count in sorted(label_statistics.items()):
    percentage = (count / total_count) * 100
    name = function_point_names.get(label, label)
    print(f"{label} ({name}): {count}个 ({percentage:.2f}%)")

print(f"\n总计: {total_count}个功能点组件")

# 输出按数量排序
print("\n2. 按数量排序：")
print("-" * 30)
sorted_by_count = sorted(label_statistics.items(), key=lambda x: x[1], reverse=True)
for label, count in sorted_by_count:
    percentage = (count / total_count) * 100
    name = function_point_names.get(label, label)
    print(f"{label} ({name}): {count}个 ({percentage:.2f}%)")

# 输出功能点计算详情
print("\n3. 功能点计算详情：")
print("-" * 30)
print(f"{'类型':<6} {'名称':<12} {'数量':<6} {'权重':<6} {'功能点':<8}")
print("-" * 45)

total_calculated_fp = 0
for label in sorted(fp_details.keys()):
    details = fp_details[label]
    print(f"{label:<6} {details['name']:<12} {details['count']:<6} {details['weight']:<6} {details['function_points']:<8}")
    total_calculated_fp += details['function_points']

print("-" * 45)
print(f"{'总计':<6} {'':<12} {total_count:<6} {'':<6} {total_calculated_fp:<8}")

# 输出最终汇总
print("\n4. 最终汇总：")
print("=" * 30)
print(f"总功能点组件数量: {total_count}个")
print(f"总功能点数(UFP): {total_calculated_fp}点")
print(f"平均每个组件功能点: {total_calculated_fp/total_count:.2f}点")

# 输出各类型占比
print(f"\n5. 各类型占比分析：")
print("-" * 30)
for label in sorted(fp_details.keys()):
    details = fp_details[label]
    count_percentage = (details['count'] / total_count) * 100
    fp_percentage = (details['function_points'] / total_calculated_fp) * 100
    print(f"{details['name']}: 数量占比 {count_percentage:.1f}%, 功能点占比 {fp_percentage:.1f}%")

print("=" * 50)