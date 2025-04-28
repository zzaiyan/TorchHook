# TorchHook

TorchHook 是一个用于管理 PyTorch 模型 Hook 的库，提供了便捷的接口来捕获特征图和调试模型。

## 安装

```bash
pip install torchhook
```

## 使用示例

```python
import torch
import torch.nn as nn
from torchhook import HookManager

# 定义一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 30 * 30, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 初始化模型和 HookManager
model = MyModel()
hook_manager = HookManager(model)

# 使用 layer_name 注册 hooks（推荐，简单易用）
hook_manager.register_forward_hook(layer_name="conv1")

# 使用 layer 对象注册 hooks（自动命名为：类名+序号）
hook_manager.register_forward_hook(layer=model.relu)

# 使用自定义名称注册 hooks（适用于调试时区分不同 hooks）
hook_manager.register_forward_hook('CustomName', layer=model.fc)

# 运行模型
for _ in range(5):
    # 生成随机输入数据
    input_tensor = torch.randn(2, 3, 32, 32)
    output = model(input_tensor)

# 打印 HookManager 信息
print(hook_manager)
print("Current keys:", hook_manager.get_keys())  # 获取所有注册的 hooks 名称

# 获取中间结果（特征图）
print("\nconv1:", hook_manager.get_features('conv1')[0].shape)  # conv1 的特征图
print("   fc:", hook_manager.get_features('CustomName')[0].shape)  # fc 的特征图

# 获取所有特征图
all_features = hook_manager.get_all()

# 将每列的特征图 concat 起来（数据量过大时可能会内存溢出）
concatenated_features = {key: torch.cat(features, dim=0) for key, features in all_features.items()}

# 计算均值和标准差
stats = {key: (torch.mean(value), torch.std(value)) for key, value in concatenated_features.items()}

# 打印结果
print("\nMean and Std of features:")
for key, (mean, std) in stats.items():
    print(f"Layer: {key}, Mean: {mean.item():.4f}, Std: {std.item():.4f}")

# 清理 hooks 和特征图
hook_manager.clear_hooks()
hook_manager.clear_features()
```

输出示例：
```sh
Model: MyModel
Layer Name                    Feature Count       Feature Shape                 
--------------------------------------------------------------------------------
conv1                         5                   (2, 16, 30, 30)               
ReLU_0                        5                   (2, 16, 30, 30)               
CustomName                    5                   (2, 10)                       
--------------------------------------------------------------------------------
Current keys: ['conv1', 'ReLU_0', 'CustomName']

conv1: torch.Size([2, 16, 30, 30])
   fc: torch.Size([2, 10])

Mean and Std of features:
Layer: conv1, Mean: -0.0060, Std: 0.5978
Layer: ReLU_0, Mean: 0.2344, Std: 0.3463
Layer: CustomName, Mean: 0.0245, Std: 0.2332
```

## 许可证

MIT License