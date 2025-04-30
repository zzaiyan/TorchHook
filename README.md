![TorchHook Logo](assets/logo.jpg)

# TorchHook

TorchHook is a library for managing PyTorch model hooks, providing convenient interfaces to capture feature maps and debug models.

## Features
- **Easy Hook Management**: Simplify the process of registering and managing hooks in PyTorch models.
- **Feature Map Extraction**: Capture intermediate feature maps for analysis and debugging.
- **Customizable**: Support for custom hook names and flexible usage.

## Installation

```bash
pip install torchhook
```

## Usage Example

```python
import torch
import torch.nn as nn
from torchhook import HookManager

# Define a simple model
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

# Initialize model and HookManager
model = MyModel()
hook_manager = HookManager(model)

# Register hooks using layer_name (recommended for simplicity)
hook_manager.register_forward_hook(layer_name="conv1")

# Register hooks using layer object (automatically named as: ClassName+Index)
hook_manager.register_forward_hook(layer=model.relu)

# Register hooks with a custom name (useful for distinguishing hooks when debugging)
hook_manager.register_forward_hook('CustomName', layer=model.fc)

# Run the model
for _ in range(5):
    # Generate random input data
    input_tensor = torch.randn(2, 3, 32, 32)
    output = model(input_tensor)

# Print HookManager information
print(hook_manager)
print("Current keys:", hook_manager.get_keys())  # Get all registered hook names

# Get intermediate results (feature maps)
print("\nconv1:", hook_manager.get_features('conv1')[0].shape)  # Feature map of conv1
print("   fc:", hook_manager.get_features('CustomName')[0].shape)  # Feature map of fc

# Get all feature maps
all_features = hook_manager.get_all()

# Concatenate feature maps for each layer (may cause memory overflow if data is too large)
concatenated_features = {key: torch.cat(features, dim=0) for key, features in all_features.items()}

# Compute mean and standard deviation
stats = {key: (torch.mean(value), torch.std(value)) for key, value in concatenated_features.items()}

# Print results
print("\nMean and Std of features:")
for key, (mean, std) in stats.items():
    print(f"Layer: {key}, Mean: {mean.item():.4f}, Std: {std.item():.4f}")

# Clear hooks and features
hook_manager.clear_hooks()
hook_manager.clear_features()
```

Example Output:
```sh
Model: MyModel | Total Parameters: 144.46 K
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
Layer: conv1, Mean: -0.0460, Std: 0.5873
Layer: ReLU_0, Mean: 0.2116, Std: 0.3276
Layer: CustomName, Mean: -0.0596, Std: 0.2248
```

## License

MIT License