![TorchHook Logo](assets/logo.jpg)

# TorchHook

[![PyPI version](https://badge.fury.io/py/torchhook.svg)](https://badge.fury.io/py/torchhook)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/torchhook.svg)](https://pypi.org/project/torchhook/)
[![Python Version](https://img.shields.io/pypi/pyversions/torchhook.svg)](https://pypi.org/project/torchhook/)

[English Blog](./BLOG.md) | [中文博客](./BLOG_CN.md) | [中文文档](./README_CN.md)

TorchHook is a lightweight, easy-to-use Python library designed to simplify the process of extracting intermediate features from PyTorch models. It provides a clean API to manage PyTorch hooks for capturing layer outputs without the boilerplate code.

## Key Features

- **Easy Hook Registration**: Quickly register hooks for desired model layers by name or object.
- **Flexible Feature Extraction**: Retrieve captured features easily.
- **Customizable**: Define custom hook logic or output transformations.
- **Resource Management**: Automatic cleanup of registered hooks.

## Installation

```bash
pip install torchhook
```
Or install from the local source:
```bash
git clone https://github.com/zzaiyan/TorchHook.git
cd TorchHook
pip install .
```

## Quick Start

```python
import torch
import torchvision.models as models
from torchhook import HookManager

# 1. Load your model
model = models.resnet18()
model.eval()

# 2. Initialize HookManager
hook_manager = HookManager(model, max_size=1) # Keep only the latest feature per hook

# 3. Register layers
hook_manager.add(layer_name='conv1')
hook_manager.add(layer_name='layer4.1.relu')
hook_manager.add(layer_name='fully_connected', layer=model.fc) # Optional: pass layer object

# 4. Forward pass
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

# 5. Get features
features_conv1 = hook_manager.get('conv1')
features_relu = hook_manager.get('layer4.1.relu')
all_features = hook_manager.get_all() # Get all features as a dict

print(f"Conv1 feature shape: {features_conv1[0].shape}")
print(f"Layer 4.1 ReLU feature shape: {features_relu[0].shape}")

# 6. Summary (Optional)
hook_manager.summary()

# 7. Clean up hooks (Important!)
hook_manager.clear_hooks()
```

For advanced usage like custom hooks and output transformations, please refer to the blog posts: [English](./BLOG.md) | [中文](./BLOG_CN.md)
