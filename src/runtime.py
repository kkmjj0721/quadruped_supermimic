# Copyright (c) 2026-2027 zh
"""内容：配置训练入口的依赖弃用提示和 CUDA 编译数学库路径。"""

import os
from pathlib import Path
import warnings


def configure_dependency_warnings() -> None:
    """保留部署所需 TorchScript，仅处理当前依赖版本的确切弃用提示。"""
    warnings.filterwarnings(
        "ignore",
        message=r"^`torch\.jit\.script` is deprecated\. Please switch to `torch\.compile` or `torch\.export`\.$",
        category=FutureWarning,
        module=r"^torch\.jit\._script$",
    )


def configure_cuda_libdevice() -> None:
    """保留显式数学库配置，否则使用标准 CUDA 安装目录。"""
    if "TRITON_LIBDEVICE_PATH" not in os.environ:
        os.environ["TRITON_LIBDEVICE_PATH"] = str(Path("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc").resolve())
