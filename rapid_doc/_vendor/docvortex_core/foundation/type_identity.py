"""保持搬迁后类型的历史导入身份及可独立解析的注解。"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import get_type_hints


def preserve_type_module(cls: type, module: str) -> None:
    """先在定义模块解析注解，再保留旧 pickle 路径，避免依赖尚未导入的门面。"""
    annotations = get_type_hints(cls)
    cls.__annotations__ = annotations
    if is_dataclass(cls):
        for field in fields(cls):
            if field.name in annotations:
                field.type = annotations[field.name]
    cls.__module__ = module


__all__ = ["preserve_type_module"]
