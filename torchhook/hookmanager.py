# -*- coding: utf-8 -*-
"""
# Author: Zaiyan Zhang
# Email: 1@zzaiyan.com
#
# Copyright (c) 2025. All rights reserved.
# This code is licensed under the MIT License.
# For more details, see the LICENSE file in the project root.
"""


import torch
from typing import Dict, List, Union, Optional, Callable


class HookManager:
    def __init__(self, model: torch.nn.Module, last_only: bool = False):
        """
        初始化 HookManager。

        参数:
        - model (torch.nn.Module): 需要添加 hook 的 PyTorch 模型。
        - last_only (bool): 是否只保存最后一个特征图。如果为 False，则保存所有特征图。

        异常:
        - TypeError: 如果 model 不是 torch.nn.Module 实例，或 last_only 不是布尔值。
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("'model' must be an instance of torch.nn.Module.")
        if not isinstance(last_only, bool):
            raise TypeError("'last_only' must be a boolean.")

        self.model = model
        self.last_only = last_only
        self.hooks = []
        self.features: Union[Dict[str, torch.Tensor],
                             Dict[str, List[torch.Tensor]]] = {}

    def register_forward_hook(
        self,
        layer_name: Optional[str] = None,
        layer: Optional[torch.nn.Module] = None,
        custom_hook: Optional[Callable] = None,
        output_transform: Optional[Callable[[
            torch.Tensor], torch.Tensor]] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        为指定层注册 forward hook。

        参数:
        - layer_name (Optional[str]): 层的名称（可选）。如果未指定 layer，则根据 layer_name 在模型中查找层。
        - layer (Optional[torch.nn.Module]): 直接传入的 nn.Module 实例（可选）。优先使用此参数。
        - custom_hook (Optional[Callable]): 自定义 hook 函数（可选）。如果未提供，则使用默认的保存特征图的 hook。
        - output_transform (Optional[Callable]): 自定义函数，用于对 output 张量进行处理（可选）。

        返回:
        - torch.utils.hooks.RemovableHandle: 注册的 hook 的句柄，可用于移除 hook。

        异常:
        - TypeError: 如果参数类型不符合要求。
        - ValueError: 如果同时提供 custom_hook 和 output_transform，或未指定 layer_name 和 layer。
        """
        if layer_name is not None and not isinstance(layer_name, str):
            raise TypeError("'layer_name' must be a string or None.")
        if layer is not None and not isinstance(layer, torch.nn.Module):
            raise TypeError(
                "'layer' must be an instance of torch.nn.Module or None.")
        if custom_hook is not None and not callable(custom_hook):
            raise TypeError("'custom_hook' must be a callable or None.")
        if output_transform is not None and not callable(output_transform):
            raise TypeError("'output_transform' must be a callable or None.")
        if custom_hook and output_transform:
            raise ValueError(
                "Cannot provide both 'custom_hook' and 'output_transform'.")

        if layer is None:
            if layer_name is None:
                raise ValueError(
                    "Either 'layer_name' or 'layer' must be specified.")
            # 根据 layer_name 在模型中查找层
            layer = dict(self.model.named_modules()).get(layer_name)
            if layer is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found in the model.")

        # 默认的 hook 函数
        def default_callback(module, input, output):
            # 使用 layer_name 或 类名+序号 作为键
            if layer_name:
                key = layer_name
            else:
                # 获取当前层的类名和序号
                layer_class_name = layer.__class__.__name__
                layer_index = len(
                    [k for k in self.features.keys() if k.startswith(layer_class_name + '-')])
                key = f"{layer_class_name}_{layer_index}"

            def default_output_transform(output):
                return output.detach().cpu()

            # 应用自定义的 output_transform（如果提供）
            transform_fn = output_transform or default_output_transform
            processed_output = transform_fn(output)

            if self.last_only:
                self.features[key] = processed_output
            else:
                if key not in self.features:
                    self.features[key] = []
                self.features[key].append(processed_output)

        # 使用自定义 hook 或默认 hook
        hook_function = custom_hook or default_callback

        # 注册 hook 并保存句柄
        hook = layer.register_forward_hook(hook_function)
        self.hooks.append(hook)

        return hook

    def add(
        self,
        layer_name: Optional[str] = None,
        layer: Optional[torch.nn.Module] = None,
        custom_hook: Optional[Callable] = None,
        output_transform: Optional[Callable[[
            torch.Tensor], torch.Tensor]] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        (alias of register_forward_hook) 为指定层注册 forward hook。

        参数:
        - layer_name (Optional[str]): 层的名称（可选）。如果未指定 layer，则根据 layer_name 在模型中查找层。
        - layer (Optional[torch.nn.Module]): 直接传入的 nn.Module 实例（可选）。优先使用此参数。
        - custom_hook (Optional[Callable]): 自定义 hook 函数（可选）。如果未提供，则使用默认的保存特征图的 hook。
        - output_transform (Optional[Callable]): 自定义函数，用于对 output 张量进行处理（可选）。

        返回:
        - torch.utils.hooks.RemovableHandle: 注册的 hook 的句柄，可用于移除 hook。

        异常:
        - TypeError: 如果参数类型不符合要求。
        - ValueError: 如果同时提供 custom_hook 和 output_transform，或未指定 layer_name 和 layer。
        """
        return self.register_forward_hook(
            layer_name=layer_name,
            layer=layer,
            custom_hook=custom_hook,
            output_transform=output_transform
        )

    def get_features(self, key: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        获取指定层的特征图。

        参数:
        - key (str): 层的名称（layer_name）或唯一标识符。

        返回:
        - Union[torch.Tensor, List[torch.Tensor]]: 如果 last_only=True，返回最后一个特征图 (torch.Tensor)；
          如果 last_only=False，返回所有特征图 (List[torch.Tensor])。

        异常:
        - TypeError: 如果 key 不是字符串。
        - ValueError: 如果指定的 key 不存在。
        """
        if not isinstance(key, str):
            raise TypeError("'key' must be a string.")
        if key not in self.features:
            raise ValueError(f"No features captured for layer '{key}'.")

        return self.features[key]

    def get_all(self) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        获取所有捕获的特征图。

        返回:
        - Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: 一个字典，键为层名称或唯一标识符，值为对应的特征图。
        """
        return self.features

    def get_keys(self) -> List[str]:
        """
        获取所有捕获特征图的键。

        返回:
        - List[str]: 一个列表，包含所有捕获特征图的键。
        """
        return list(self.features.keys())

    def remove(self, key: str):
        """
        删除指定 key 对应的 hook 和存储的特征图。

        参数:
        - key (str): 要删除的层的名称或唯一标识符。

        异常:
        - ValueError: 如果指定的 key 不存在。
        """
        if key not in self.features:
            raise ValueError(f"No features or hooks found for key '{key}'.")

        # 删除特征图
        del self.features[key]

        # 删除对应的 hook
        for hook in self.hooks:
            if hasattr(hook, 'layer_name') and hook.layer_name == key:
                hook.remove()
                self.hooks.remove(hook)
                break

    def clear_hooks(self):
        """
        移除所有已注册的 hooks。

        异常:
        - TypeError: 如果 hooks 不是列表，或列表中的元素没有 remove 方法。
        """
        if not isinstance(self.hooks, list):
            raise TypeError("'hooks' must be a list of hook handles.")
        for hook in self.hooks:
            if not hasattr(hook, "remove") or not callable(hook.remove):
                raise TypeError(
                    "Each hook must have a callable 'remove' method.")
            hook.remove()
        self.hooks.clear()

    def clear_features(self):
        """
        清空已捕获的特征图。

        异常:
        - TypeError: 如果 features 不是字典。
        """
        if not isinstance(self.features, dict):
            raise TypeError("'features' must be a dictionary.")
        self.features.clear()

    def summary(self):
        """
        打印当前模型的名称、已注册的 hook 信息、特征数量和特征图形状。
        """
        print(self.__str__())

    def __str__(self):
        """
        重载 __str__ 方法，优化 print(hook_manager) 的输出。

        返回:
        - str: 包含模型名称、已注册的 hook 信息、特征数量和特征图形状的字符串。
        """
        if not self.hooks:
            # 如果没有注册任何 hook
            return f"Model: {self.model.__class__.__name__}\nNo hooks have been registered."

        if not self.features:
            # 如果没有捕获到任何特征图
            return f"Model: {self.model.__class__.__name__}\nHooks are registered, but no features have been captured yet."

        # 如果有特征图，正常显示信息
        output = [f"Model: {self.model.__class__.__name__}"]
        output.append(
            f"{'Layer Name':<30}{'Feature Count':<20}{'Feature Shape':<30}")
        output.append("-" * 80)

        for key, value in self.features.items():
            if isinstance(value, torch.Tensor):
                # 如果只保存最后一个特征图
                feature_count = 1
                feature_shape = tuple(value.shape)
            elif isinstance(value, list):
                # 如果保存所有特征图
                feature_count = len(value)
                feature_shape = tuple(
                    value[0].shape) if feature_count > 0 else "N/A"
            else:
                feature_count = 0
                feature_shape = "N/A"

            output.append(
                f"{key:<30}{feature_count:<20}{str(feature_shape):<30}")

        output.append("-" * 80)

        return "\n".join(output)
