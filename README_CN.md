![TorchHook Logo](assets/logo.jpg)

# TorchHook

[![PyPI version](https://badge.fury.io/py/torchhook.svg)](https://badge.fury.io/py/torchhook)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/torchhook.svg)](https://pypi.org/project/torchhook/)
[![Python Version](https://img.shields.io/pypi/pyversions/torchhook.svg)](https://pypi.org/project/torchhook/)

[English Blog](./BLOG.md) | [中文博客](./BLOG_CN.md) | [English Readme](./README.md)

TorchHook 是一个轻量级、易于使用的 Python 库，旨在简化从 PyTorch 模型中提取中间特征的过程。它提供简洁的 API 来管理 PyTorch 的钩子（Hooks），以便捕获层输出，而无需编写重复的模板代码。

## 主要特性

- **轻松注册 Hook**：通过层名称或层对象快速为所需模型层注册钩子。
- **灵活提取特征**：方便地检索捕获到的特征。
- **高度可定制**：可定义自定义的钩子逻辑或输出转换函数。
- **自动资源管理**：自动清理已注册的钩子。

## 安装

```bash
pip install torchhook
```
或者从本地源码安装：
```bash
git clone https://github.com/zzaiyan/TorchHook.git
cd TorchHook
pip install .
```

## 快速上手

```python
import torch
import torchvision.models as models
# 导入我们的库 TorchHook
from torchhook import HookManager

# 1. 加载你的模型
model = models.resnet18()
model.eval() # 设置为评估模式

# 2. 初始化 HookManager
# max_size=1 表示每个 hook 只保留最新的特征图
hook_manager = HookManager(model, max_size=1)

# 3. 注册你感兴趣的层
# 通过层名称注册
hook_manager.add(layer_name='conv1')
hook_manager.add(layer_name='layer4.1.relu')
# 或者直接传入层对象 (建议提供名称)
hook_manager.add(layer_name='fully_connected', layer=model.fc)

# 4. 执行模型的前向传播
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

# 5. 获取特征
features_conv1 = hook_manager.get('conv1') # 获取 'conv1' 的特征列表
features_relu = hook_manager.get('layer4.1.relu') # 获取 'layer4.1.relu' 的特征列表
all_features = hook_manager.get_all() # 获取包含所有捕获的特征的字典

print(f"Conv1 feature shape: {features_conv1[0].shape}")
print(f"Layer 4.1 ReLU feature shape: {features_relu[0].shape}")

# 6. 查看 Hook 状态总结 (可选)
hook_manager.summary()

# 7. 清理 Hook（重要！）
hook_manager.clear_hooks()
```

关于自定义钩子和输出转换等进阶用法，请参阅博客文章：[English](./BLOG.md) | [中文](./BLOG_CN.md)。
