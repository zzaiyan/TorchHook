# TorchHook: Easily Access PyTorch Model Intermediate Outputs to Accelerate Your Deep Learning Research

In deep learning model research, we often need to delve into the internal workings of models. Whether it's visualizing feature maps, checking neuron activation states, analyzing attention distributions, or generating heatmaps to explain model decisions, obtaining intermediate layer outputs is a crucial step. However, manually managing PyTorch's Hooks API can be tedious and error-prone, especially when dealing with complex models or needing to capture outputs from multiple layers.

To address this, we are pleased to introduce **TorchHook** – a lightweight, easy-to-use Python library designed to simplify the process of extracting intermediate features from PyTorch models.

## Why Choose TorchHook?

PyTorch's built-in `register_forward_hook` is powerful, but using it directly often requires writing repetitive boilerplate code to handle feature storage, management, and cleanup. TorchHook encapsulates this functionality, providing a cleaner, more intuitive API that allows you to:

- **Register Hooks Easily**: Quickly register desired model layers with simple API calls, eliminating the need to manually manage complex hook logic.
- **Extract Features Flexibly**: Supports extracting intermediate features by layer name or layer object, and even allows customizing the output logic for each hook.
- **Highly Customizable**: Enables users to define custom hooks or output transformation functions to meet specific needs.
- **Reliable Resource Management**: Provides an automated hook cleanup mechanism to prevent resource leaks and potential performance issues.

## Quick Start

Suppose you have a PyTorch model and want to get the output feature maps from layers named `conv1` and `layer4.2.relu`. Using TorchHook is straightforward:

```python
import torch
import torchvision.models as models
# Import our library TorchHook
from torchhook import HookManager

# 1. Load your model
model = models.resnet18()
model.eval() # Set to evaluation mode

# 2. Initialize HookManager
# max_size=1 means each hook only keeps the latest feature map
hook_manager = HookManager(model, max_size=1)

# 3. Register the layers you are interested in
# Register by layer name
# Layer names can be obtained from dict(model.named_modules()).keys()
hook_manager.register_forward_hook(layer_name='conv1')
# You can also use the alias 'add'
hook_manager.add(layer_name='layer4.1.relu')
# Or pass the layer object directly
target_layer = model.fc
hook_manager.add(layer_name='fully_connected', layer=target_layer) # Providing a name is recommended

# 4. Perform the model's forward pass
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)

# 5. Get the features
features_conv1 = hook_manager.get_features('conv1') # Get the list of features for 'conv1'
features_relu = hook_manager.get_features('layer4.1.relu') # Get the list of features for 'layer4.1.relu'
all_features = hook_manager.get_all() # Get a dictionary containing all captured features

print(f"Conv1 feature shape: {features_conv1[0].shape}")
print(f"Layer 4.1 ReLU feature shape: {features_relu[0].shape}")

# 6. View Hook status summary
hook_manager.summary()
# Or print(hook_manager)

# 7. Clean up hooks (Important!)
hook_manager.clear_hooks()
```

Example Output:
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

## Advanced Usage

### Custom Hook Logic

If you need more complex logic (e.g., saving features only when a specific condition is met, or saving modified features), you can use the `custom_hook` parameter:

```python
def my_custom_hook(module, input, output):
    # Example: Only save the mean of positive values in the output tensor
    if isinstance(output, torch.Tensor):
        positive_mean = output[output > 0].mean()
        # Note: custom_hook needs to return a Tensor or None
        # Here we return a Tensor containing a single value
        return torch.tensor([positive_mean])
    return None # Return None if not saving

hook_manager.add(layer_name='layer1.0.conv1', custom_hook=my_custom_hook)

# ... Perform forward pass ...

custom_features = hook_manager.get_features('layer1.0.conv1')
if custom_features:
    print(f"Custom feature (positive mean): {custom_features[0].item()}")
```

### Output Transformation

If you only need to perform a simple transformation on the output tensor (e.g., apply `softmax` or change shape) before storing it, use `output_transform`:

```python
def apply_softmax(output_tensor):
    # Assume the output is logits
    return torch.softmax(output_tensor, dim=-1).detach().cpu()

# Assume the 'fc' layer outputs logits
hook_manager.add(layer_name='fc', output_transform=apply_softmax)

# ... Perform forward pass ...

softmax_output = hook_manager.get_features('fc')
if softmax_output:
    print(f"Softmax output shape: {softmax_output[0].shape}")

```

## Installation

You can install TorchHook via pip (or installed from source):

```bash
pip install torchhook
```
Or install from the local source:
```bash
git clone https://github.com/zzaiyan/TorchHook.git
cd TorchHook
pip install .
```

## Summary

TorchHook provides PyTorch users with a concise and efficient tool for capturing and managing intermediate layer outputs of models. Whether you need to visualize feature maps, debug model behavior, or perform deeper model analysis, TorchHook can save you time and effort, allowing you to focus more on your core research work.

We encourage you to try TorchHook and welcome any feedback and contributions!

Project Repository: [https://github.com/zzaiyan/TorchHook](https://github.com/zzaiyan/TorchHook)
