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
from typing import Dict


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型的参数总量。

    参数:
    - model (torch.nn.Module): PyTorch 模型。
    - trainable_only (bool): 是否只统计可训练参数。

    返回:
    - int: 模型的参数总量。
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("'model' must be an instance of torch.nn.Module.")

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_parameter_count(count: int, precision: int = 2) -> str:
    """
    格式化参数量为可读字符串。

    参数:
    - count (int): 参数总量。
    - precision (int): 保留的小数位数。

    返回:
    - str: 格式化后的参数量字符串。
    """
    units = ['', 'K', 'M', 'G', 'T']
    unit_idx = 0
    while count >= 1000 and unit_idx < len(units) - 1:
        count /= 1000
        unit_idx += 1
    return f"{count:.{precision}f} {units[unit_idx]}"


def get_layerwise_parameter_count(model: torch.nn.Module, max_depth: int = -1, trainable_only: bool = False) -> Dict[str, int]:
    """
    使用深度优先搜索 (DFS) 统计每一层的参数量，跳过顶层模型。

    参数:
    - model (torch.nn.Module): PyTorch 模型。
    - max_depth (int): 最大深度，-1 表示不限制深度。
    - trainable_only (bool): 是否只统计可训练参数。

    返回:
    - Dict[str, int]: 每一层的参数量字典，键为层名称，值为参数量。
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("'model' must be an instance of torch.nn.Module.")

    def dfs(layer, depth, prefix):
        if max_depth != -1 and depth > max_depth:
            return {}

        # 计算当前层的参数
        if trainable_only:
            params = {prefix: sum(p.numel() for p in layer.parameters(
                recurse=False) if p.requires_grad)}
        else:
            params = {prefix: sum(p.numel()
                                  for p in layer.parameters(recurse=False))}

        # 递归处理子层
        for child_name, child in layer.named_children():
            child_params = dfs(child, depth + 1, f"{prefix}.{child_name}")
            params.update(child_params)

        return params

    # 从顶层模型的子层开始遍历
    all_params = {}
    for name, child in model.named_children():
        all_params.update(dfs(child, 0, name))

    return all_params


def model_summary(
        model: torch.nn.Module,
        max_depth: int = -1,
        show_zero_params: bool = True):
    """
    打印模型的摘要信息，包括模型名称、总参数量、可训练参数量以及每一层的参数量。

    参数:
    - model (torch.nn.Module): PyTorch 模型。
    - max_depth (int): 统计层级参数时使用的最大深度，-1 表示不限制深度。
    - show_zero_params (bool): 是否显示参数量为 0 的层。
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("'model' must be an instance of torch.nn.Module.")

    model_name = model.__class__.__name__
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    layer_params = get_layerwise_parameter_count(model, max_depth=max_depth)
    trainable_layer_params = get_layerwise_parameter_count(
        model, max_depth=max_depth, trainable_only=True)

    # 格式化输出
    summary_str = []
    summary_str.append("=" * 80)
    summary_str.append(f"Model Summary: {model_name}")
    summary_str.append("-" * 80)
    summary_str.append(
        f"Total Parameters: {format_parameter_count(total_params)}")
    summary_str.append(
        f"Trainable Parameters: {format_parameter_count(trainable_params)}")
    summary_str.append(
        f"Non-trainable Parameters: {format_parameter_count(total_params - trainable_params)}")
    summary_str.append("-" * 80)
    summary_str.append(
        f"{'Layer Name':<40} {'Total Params':<20} {'Trainable Params':<20}")
    summary_str.append("=" * 80)

    indent_char = "  "
    for name, count in layer_params.items():
        if not show_zero_params and count == 0:
            continue

        trainable_count = trainable_layer_params.get(name, 0)
        # 计算缩进
        depth = name.count('.')
        indent = indent_char * depth
        display_name = f"{indent}{name}"

        summary_str.append(
            f"{display_name:<40} {format_parameter_count(count):<20} {format_parameter_count(trainable_count):<20}")

    summary_str.append("=" * 80)
    print("\n".join(summary_str))
