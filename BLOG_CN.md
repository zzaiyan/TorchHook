# TorchHook：轻松获取 PyTorch 模型中间输出，加速你的深度学习研究

在基于深度学习模型的科研工作中，我们经常需要深入了解模型的内部工作机制。无论是为了可视化特征图、检查神经元的激活状态、分析注意力分布，还是生成热力图来解释模型决策，获取模型的中间层输出都是一个至关重要的步骤。然而，手动管理 PyTorch 的 Hooks API 可能既繁琐又容易出错，尤其是在处理复杂模型或需要捕获多个层输出时。

为了解决这个问题，我们很高兴向你介绍 **TorchHook** —— 一个轻量级、易于使用的 Python 库，旨在简化从 PyTorch 模型中提取中间特征的过程。

## 为什么选择 TorchHook？

PyTorch 内置的 `register_forward_hook` 功能强大，但直接使用它通常需要编写重复的模板代码来处理特征的存储、管理和清理。TorchHook 在此基础上进行了封装，提供了更简洁、更直观的 API，让你能够：

- **轻松注册 Hook**：通过简单的 API 调用即可快速注册所需的模型层，无需手动管理复杂的 Hook 逻辑。
- **灵活提取特征**：支持按层名称或层对象提取中间特征，甚至可以为每个 Hook 定制输出逻辑。
- **高度可定制**：允许用户定义自定义 Hook 或输出转换函数，以满足特定需求。
- **可靠资源管理**：提供自动化的 Hook 清理机制，避免资源泄漏和潜在的性能问题。

## 快速上手

假设你有一个 PyTorch 模型，并希望获取其中名为 `conv1` 和 `layer4.2.relu` 的层的输出特征图。使用 TorchHook 非常简单：

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
# 层的名称可从 dict(model.named_modules()).keys() 中获取
hook_manager.register_forward_hook(layer_name='conv1')
# 也可以使用别名 add
hook_manager.add(layer_name='layer4.1.relu')
# 或者直接传入层对象
target_layer = model.fc
hook_manager.add(layer_name='fully_connected', layer=target_layer) # 建议提供名称

# 4. 执行模型的前向传播
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

# 5. 获取特征
features_conv1 = hook_manager.get_features('conv1') # 获取 'conv1' 的特征列表
features_relu = hook_manager.get_features('layer4.1.relu') # 获取 'layer4.1.relu' 的特征列表
all_features = hook_manager.get_all() # 获取包含所有捕获的特征的字典

print(f"Conv1 feature shape: {features_conv1[0].shape}")
print(f"Layer 4.1 ReLU feature shape: {features_relu[0].shape}")

# 6. 查看 Hook 状态总结
hook_manager.summary()
# 或者 print(hook_manager)

# 7. 清理 Hook（重要！）
hook_manager.clear_hooks()
```

输出示例：
```sh
Conv1 feature shape: torch.Size([1, 64, 112, 112])
Layer 4.1 ReLU feature shape: torch.Size([1, 512, 7, 7])
Model: ResNet | Total Parameters: 11.69 M
Registered Hooks: 3 (max_size=1)
--------------------------------------------------------------------------------
Captured Features Summary:
Layer Key                     Feature Count       Feature Shape                 
--------------------------------------------------------------------------------
conv1                         1                   (1, 64, 112, 112)             
layer4.1.relu                 1                   (1, 512, 7, 7)                
fully_connected               1                   (1, 1000)                     
--------------------------------------------------------------------------------
```

## 进阶用法

### 自定义 Hook 逻辑

如果你需要更复杂的逻辑（例如，只在满足特定条件时保存特征，或者保存修改后的特征），可以使用 `custom_hook` 参数：

```python
def my_custom_hook(module, input, output):
    # 示例：只保存输出张量中正值的平均值
    if isinstance(output, torch.Tensor):
        positive_mean = output[output > 0].mean()
        # 注意：custom_hook 需要返回一个 Tensor 或 None
        # 这里我们返回一个包含单个值的 Tensor
        return torch.tensor([positive_mean])
    return None # 如果不保存，返回 None

hook_manager.add(layer_name='layer1.0.conv1', custom_hook=my_custom_hook)

# ... 执行前向传播 ...

custom_features = hook_manager.get_features('layer1.0.conv1')
if custom_features:
    print(f"Custom feature (positive mean): {custom_features[0].item()}")
```

### 输出转换

如果你只想对输出张量进行简单的转换（例如，应用 `softmax` 或改变形状）后再存储，可以使用 `output_transform`：

```python
def apply_softmax(output_tensor):
    # 假设输出是 logits
    return torch.softmax(output_tensor, dim=-1).detach().cpu()

# 假设 'fc' 层输出 logits
hook_manager.add(layer_name='fc', output_transform=apply_softmax)

# ... 执行前向传播 ...

softmax_output = hook_manager.get_features('fc')
if softmax_output:
    print(f"Softmax output shape: {softmax_output[0].shape}")

```

## 安装

你可以通过 pip 安装 TorchHook（或从源码安装）：

```bash
pip install torchhook
```
或者从本地源码安装：
```bash
git clone https://github.com/zzaiyan/TorchHook.git
cd TorchHook
pip install .
```

## 总结

TorchHook 为 PyTorch 用户提供了一个简洁高效的工具，用于捕获和管理模型的中间层输出。无论你是需要可视化特征图、调试模型行为，还是进行更深入的模型分析，TorchHook 都能帮你节省时间和精力，让你更专注于核心的研究工作。

我们鼓励你尝试 TorchHook，并欢迎任何反馈和贡献！

项目仓库地址：[https://github.com/zzaiyan/TorchHook](https://github.com/zzaiyan/TorchHook)
