# Copyright (c) 2026-2027 zh
"""内容：配置训练入口的已知依赖弃用提示。"""

import warnings


def configure_dependency_warnings() -> None:
    """保留部署所需 TorchScript，仅处理当前依赖版本的确切弃用提示。"""
    warnings.filterwarnings(
        "ignore",
        message=r"^`torch\.jit\.script` is deprecated\. Please switch to `torch\.compile` or `torch\.export`\.$",
        category=FutureWarning,
        module=r"^torch\.jit\._script$",
    )
