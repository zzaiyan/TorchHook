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
import warnings
from typing import Dict, List, Union, Optional, Callable, Any
from .utils import format_parameter_count, count_parameters


class HookManager:
    def __init__(self, model: torch.nn.Module, max_size: int = -1):
        """
        初始化 HookManager。

        参数:
        - model (torch.nn.Module): 需要添加 hook 的 PyTorch 模型。
        - max_size (int): 每个 hook 最多保存的特征图数量。默认为 -1，表示不限制数量。
                          如果为 1，则只保留最新的特征图。

        异常:
        - TypeError: 如果 model 不是 torch.nn.Module 实例，或 max_size 不是整数。
        - ValueError: 如果 max_size 小于 -1。
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("'model' must be an instance of torch.nn.Module.")
        if not isinstance(max_size, int):
            raise TypeError("'max_size' must be an integer.")
        if max_size < -1:
            raise ValueError("'max_size' cannot be less than -1.")

        self.model = model
        self.max_size = max_size
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        # 特征始终存储在列表中
        self.features: Dict[str, List[torch.Tensor]] = {}

    def register_forward_hook(
        self,
        layer_name: Optional[str] = None,
        layer: Optional[torch.nn.Module] = None,
        custom_hook: Optional[Callable[[
            torch.nn.Module, Any, Any], Optional[torch.Tensor]]] = None,  # Modified signature hint
        output_transform: Optional[Callable[[
            torch.Tensor], torch.Tensor]] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        为指定层注册 forward hook。

        参数:
        - layer_name (Optional[str]): 层的名称（可选）。如果未指定 layer，则根据 layer_name 在模型中查找层。此名称将用作存储特征和 hook 的键。
        - layer (Optional[torch.nn.Module]): 直接传入的 nnModule 实例（可选）。优先使用此参数。
        - custom_hook (Optional[Callable]): 自定义 hook 函数（可选）。
            签名应为 `hook(module, input, output) -> Optional[torch.Tensor]`。
            它应返回要存储的 torch.Tensor，或者返回 None 表示不存储任何内容。
            如果提供了 custom_hook，则忽略 output_transform。
        - output_transform (Optional[Callable]): 自定义函数，用于对 output 张量进行处理（可选）。
            仅在未提供 custom_hook 时生效。签名应为 `transform(output) -> torch.Tensor`。

        返回:
        - torch.utils.hooks.RemovableHandle: 注册的 hook 的句柄，可用于移除 hook。

        异常:
        - TypeError: 如果参数类型不符合要求。
        - ValueError: 如果同时提供 custom_hook 和 output_transform（现在允许，但 custom_hook 优先），
                      或未指定 layer_name 和 layer，或指定的 layer_name 已存在。
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

        if layer is None:
            if layer_name is None:
                raise ValueError(
                    "Either 'layer_name' or 'layer' must be specified.")
            # 根据 layer_name 在模型中查找层
            _layer = dict(self.model.named_modules()).get(layer_name)
            if _layer is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found in the model.")
            layer = _layer  # Assign found layer back to layer variable
        else:
            # 如果直接提供了 layer，需要确定一个 key
            if layer_name is None:
                # 生成一个唯一的 key
                layer_class_name = layer.__class__.__name__
                layer_index = len(
                    [k for k in self.hooks.keys() if k.startswith(layer_class_name + '_')])
                layer_name = f"{layer_class_name}_{layer_index}"

        # 检查 key 是否已存在
        if layer_name in self.hooks:
            raise ValueError(f"Hook for key '{layer_name}' already exists.")

        # 确定最终的 key
        key = layer_name

        # 始终初始化特征列表
        self.features[key] = []

        # 定义默认的 output transform
        def default_output_transform(output: torch.Tensor) -> torch.Tensor:
            return output.detach().cpu()

        # 最终注册到 PyTorch 的 hook 回调函数
        def _final_hook_callback(module: torch.nn.Module, input: Any, output: Any):
            # 使用外部作用域的 key
            current_key = key
            tensor_to_store: Optional[torch.Tensor] = None

            if custom_hook:
                # 如果提供了自定义 hook，调用它并获取结果
                result = custom_hook(module, input, output)
                if result is not None:
                    if isinstance(result, torch.Tensor):
                        tensor_to_store = result
                    else:
                        # 使用 warnings.warn 替换 print
                        warnings.warn(
                            f"custom_hook for key '{current_key}' returned a non-Tensor value ({type(result)}). Ignoring.", stacklevel=2)
                        # 或者 raise TypeError(...)
            else:
                # 如果没有自定义 hook，使用 output_transform 或默认 transform
                transform_fn = output_transform or default_output_transform
                # 确保 output 是 Tensor 或可以转换为 Tensor
                if isinstance(output, torch.Tensor):
                    tensor_to_store = transform_fn(output)
                elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    # 处理模型输出是元组或列表的情况
                    tensor_to_store = transform_fn(output[0])
                    warnings.warn(
                        f"Output for key '{current_key}' is a sequence; processing the first element. Use 'output_transform' for custom handling.", stacklevel=2)
                else:
                    warnings.warn(
                        f"Output for key '{current_key}' is not a Tensor or a sequence starting with a Tensor ({type(output)}). Cannot store.", stacklevel=2)

            # 如果获得了要存储的 Tensor
            if tensor_to_store is not None:
                # 获取当前键对应的特征列表
                feature_list = self.features[current_key]
                # 添加新特征
                feature_list.append(tensor_to_store)
                # 如果设置了 max_size 且列表长度超过限制，则移除旧特征
                if self.max_size > 0 and len(feature_list) > self.max_size:
                    # 从列表开头移除多余的元素
                    del feature_list[:-self.max_size]

        # 注册最终的回调函数
        hook = layer.register_forward_hook(_final_hook_callback)
        self.hooks[key] = hook  # Store hook handle in the dictionary

        return hook

    def add(
        self,
        layer_name: Optional[str] = None,
        layer: Optional[torch.nn.Module] = None,
        custom_hook: Optional[Callable[[
            torch.nn.Module, Any, Any], None]] = None,
        output_transform: Optional[Callable[[
            torch.Tensor], torch.Tensor]] = None
    ) -> torch.utils.hooks.RemovableHandle:
        """
        (alias of register_forward_hook) 为指定层注册 forward hook。

        参数:
        - layer_name (Optional[str]): 层的名称（可选）。如果未指定 layer，则根据 layer_name 在模型中查找层。此名称将用作存储特征和 hook 的键。
        - layer (Optional[torch.nn.Module]): 直接传入的 nnModule 实例（可选）。优先使用此参数。
        - custom_hook (Optional[Callable]): 自定义 hook 函数（可选）。如果未提供，则使用默认的保存特征图的 hook。签名应为 `hook(module, input, output)`。
        - output_transform (Optional[Callable]): 自定义函数，用于对 output 张量进行处理（可选）。仅在未使用 custom_hook 时生效。

        返回:
        - torch.utils.hooks.RemovableHandle: 注册的 hook 的句柄，可用于移除 hook。

        异常:
        - TypeError: 如果参数类型不符合要求。
        - ValueError: 如果同时提供 custom_hook 和 output_transform，或未指定 layer_name 和 layer，或指定的 layer_name 已存在。
        """
        return self.register_forward_hook(
            layer_name=layer_name,
            layer=layer,
            custom_hook=custom_hook,
            output_transform=output_transform
        )

    def get_features(self, key: str) -> List[torch.Tensor]:
        """
        获取指定层的特征图列表。

        参数:
        - key (str): 层的名称（layer_name）或唯一标识符。

        返回:
        - List[torch.Tensor]: 包含捕获的特征图的列表。列表的长度受初始化时的 `max_size` 限制。
                           如果 hook 尚未触发，则返回空列表。

        异常:
        - TypeError: 如果 key 不是字符串。
        - ValueError: 如果指定的 key 未注册 hook。
        """
        if not isinstance(key, str):
            raise TypeError("'key' must be a string.")

        # 检查 hook 是否已注册
        if key not in self.hooks:
            raise ValueError(f"No hook registered with key '{key}'.")

        # 检查特征列表是否存在（理论上总存在，因为注册时会创建）
        if key not in self.features:
            # 这表示内部状态可能不一致
            raise ValueError(
                f"Inconsistent state: Hook '{key}' registered but no feature entry found.")

        # 直接返回特征列表（可能为空）
        return self.features[key]

    def get_all(self) -> Dict[str, List[torch.Tensor]]:
        """
        获取所有捕获的特征图。

        返回:
        - Dict[str, List[torch.Tensor]]: 一个字典，键为层名称或唯一标识符，值为对应的特征图列表。
          每个列表的长度受初始化时的 `max_size` 限制。
        """
        return self.features

    def get_keys(self) -> List[str]:
        """
        获取所有捕获特征图的键。

        返回:
        - List[str]: 一个列表，包含所有捕获特征图的键。
        """
        return list(self.hooks.keys())

    def remove(self, key: str):
        """
        删除指定 key 对应的 hook 和存储的特征图。

        参数:
        - key (str): 要删除的层的名称或唯一标识符。

        异常:
        - TypeError: 如果 key 不是字符串。
        - ValueError: 如果指定的 key 不存在于 hooks 中。
        """
        if not isinstance(key, str):
            raise TypeError("'key' must be a string.")

        # 删除特征图（如果存在）
        if key in self.features:
            del self.features[key]

        # 删除 hook
        if key in self.hooks:
            hook = self.hooks[key]
            hook.remove()
            del self.hooks[key]
        else:
            # 如果 hook 不存在，但特征存在（理论上不应发生，除非手动修改了字典），也提示一下
            if key not in self.features:
                raise ValueError(f"No hook or features found for key '{key}'.")
            # If features existed but hook didn't, the feature part is already removed.
            # No exception needed here if features were removed.

    def clear_hooks(self):
        """
        移除所有已注册的 hooks，并清空特征图列表。

        异常:
        - TypeError: 如果 hooks 不是字典，或字典中的值没有 remove 方法。
        """
        if not isinstance(self.hooks, dict):
            raise TypeError("'hooks' must be a dictionary of hook handles.")
        for key, hook in list(self.hooks.items()):  # Iterate over a copy of items
            if not hasattr(hook, "remove") or not callable(hook.remove):
                # This check might be redundant if RemovableHandle is always used,
                # but kept for safety.
                raise TypeError(
                    f"Hook associated with key '{key}' does not have a callable 'remove' method.")
            hook.remove()
            # Optionally remove the key from the dictionary immediately,
            # although clear() will handle it later.
            # del self.hooks[key]
        # 清空特征图列表
        self.hooks.clear()
        self.features.clear()  # 需要保持 hooks 和 features 的一致性

    def clear_features(self):
        """
        清空所有已捕获的特征图，但保留 hook 键。
        """
        if not isinstance(self.features, dict):
            raise TypeError("'features' must be a dictionary.")
        # 遍历字典中的每个列表并清空它
        for key in self.features:
            self.features[key].clear()

    def summary(self):
        """
        打印当前模型的名称、已注册的 hook 信息、特征数量和特征图形状。
        """
        print(self.__str__())

    def __repr__(self):
        """
        重载 __repr__ 方法，优化 print(hook_manager) 的输出。

        返回:
        - str: 包含模型名称、已注册的 hook 信息、特征数量和特征图形状的字符串。
        """
        return self.__str__()

    def __str__(self):
        """
        重载 __str__ 方法，优化 print(hook_manager) 的输出。

        返回:
        - str: 包含模型名称、已注册的 hook 信息、特征数量和特征图形状的字符串。
        """
        registered_hooks_count = len(self.hooks)
        hook_keys = list(self.hooks.keys())

        model_name_line = f"Model: {self.model.__class__.__name__}"
        try:
            total_params = count_parameters(self.model)
            model_name_line += f" | Total Parameters: {format_parameter_count(total_params)}"
        except Exception:
            model_name_line += " | Total Parameters: N/A"

        if registered_hooks_count == 0:
            return f"{model_name_line}\nNo hooks have been registered."

        has_captured_features = any(
            len(lst) > 0 for lst in self.features.values())

        if not has_captured_features:
            return f"{model_name_line}\n{registered_hooks_count} hooks registered ({', '.join(hook_keys)}), but no features have been captured yet."

        output = [model_name_line]
        # 显示 max_size
        output.append(
            f"Registered Hooks: {registered_hooks_count} (max_size={self.max_size if self.max_size > 0 else 'unlimited'})")
        output.append("-" * 80)
        output.append("Captured Features Summary:")
        output.append(
            f"{'Layer Key':<30}{'Feature Count':<20}{'Feature Shape':<30}")
        output.append("-" * 80)

        triggered_hooks_count = 0
        untriggered_hooks = []

        for key in hook_keys:
            value = self.features.get(key, [])
            feature_count = len(value)
            feature_shape = tuple(
                value[0].shape) if feature_count > 0 else "N/A"

            if feature_count > 0:
                output.append(
                    f"{key:<30}{str(feature_count):<20}{str(feature_shape):<30}")
                triggered_hooks_count += 1
            else:
                untriggered_hooks.append(key)

        if untriggered_hooks:
            if triggered_hooks_count == 0:
                output = output[:3]
            else:
                output.append("-" * 80)
            output.append("Hooks Registered but Not Triggered Yet:")
            for key in untriggered_hooks:
                output.append(f"- {key}")

        output.append("-" * 80)
        return "\n".join(output)
