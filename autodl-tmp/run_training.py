#!/usr/bin/env python
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

# 创建模拟的模块
class GroupDataset:
    def __init__(self, *args, **kwargs):
        pass

def load_group_data(*args, **kwargs):
    return None

# 为模块打补丁
import sys
class MockModule:
    def __init__(self):
        self.GroupDataset = GroupDataset
        self.load_group_data = load_group_data
        
mock = MockModule()
sys.modules['data.graph_dataset'] = mock

# 现在可以运行训练脚本
import subprocess
subprocess.call("./train_all_models.sh", shell=True)
