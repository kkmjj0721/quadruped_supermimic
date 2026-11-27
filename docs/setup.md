# 安装配置

本项目使用 uv 管理独立的 `.venv`，依赖版本与下载来源记录在 `uv.lock`。

## 环境与版本

支持 Linux x86_64。CUDA 模式需要兼容 CUDA 13.0 的 NVIDIA 驱动；CPU 模式用于导入和配置开发，不代表支持 MjLab 的 GPU 训练。

| 组件 | 版本与来源 |
| --- | --- |
| Python | `.python-version` 固定为 3.11.16，由 uv 准备 |
| uv | 至少 0.8.0 |
| MjLab | `mjlab==1.6.0`，来自 PyPI |
| PyTorch | `torch==2.14.1`，使用官方 CPU 或 cu130 wheel |
| torchvision | `torchvision==0.29.1`，与 PyTorch 使用相同后端 |
| 其他依赖 | 完整版本与下载校验信息记录于 `uv.lock` |

项目算法从本地 `agents` 导入，目前仅包含 SuperMimic，见 [算法说明](../agents/README.md)。

## 安装与检查

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv 并加入 PATH，再在仓库运行：

```bash
bash setup.sh --torch-backend cu130
uv run --no-sync python -B tools/check_install.py --device cuda
```

`setup.sh` 定位自身所在仓库，然后直接执行 `uv sync --locked --extra <backend>`。uv 根据 `.python-version` 准备 Python 并同步环境；安装错误保留原始输出和退出码。脚本不安装系统驱动或 CUDA Toolkit。

检查工具验证核心依赖、MuJoCo 模型步进、PyTorch 运算、torchvision NMS 和 Warp 设备分配及数据传输。失败保留完整异常堆栈；检查不包含训练、策略效果或图形窗口验证。

```bash
bash setup.sh --help
uv run --no-sync python -B tools/check_install.py --help
```

## 后端选择

| 安装命令 | 行为 |
| --- | --- |
| `bash setup.sh` | 读取 PATH 中 `nvcc --version`；Toolkit 至少为 13.0 时选 `cu130`，没有 nvcc 或版本更低时选 `cpu` |
| `bash setup.sh --torch-backend cu130` | 使用 CUDA 13.0 wheel，适用于已有兼容驱动但没有 nvcc 的机器 |
| `bash setup.sh --torch-backend cpu` | 使用 CPU wheel |

自动选择依据 Toolkit 版本；nvcc 执行失败或输出无法解析时直接退出。预编译 PyTorch CUDA wheel 本身不要求本机安装同版本 Toolkit，只有兼容驱动的机器可显式选择 `cu130`。

`cpu` 与 `cu130` 是互斥的可选依赖组。日常命令使用 `uv run --no-sync`，避免未指定后端的自动同步改变环境。手动同步时带上后端，例如 `uv sync --locked --extra cu130`。

## 镜像、代理与离线模式

镜像、代理、超时及缓存均使用 uv 原生配置，不做测速、URL 改写或多源回退。Python 镜像通过 `UV_PYTHON_INSTALL_MIRROR` 设置，参见 [uv Python 安装说明](https://docs.astral.sh/uv/guides/install-python/)：

```bash
UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone \
  bash setup.sh --torch-backend cu130
```

普通包索引通过 `UV_DEFAULT_INDEX` 配置。由于安装使用 `--locked`，更换索引时先显式更新锁文件并审阅差异，再用同一配置同步；不会在安装阶段自动改写锁文件。PyTorch 仍使用 `pyproject.toml` 中声明的专用索引。

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv lock
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash setup.sh --torch-backend cu130
```

代理使用标准 `HTTP_PROXY`、`HTTPS_PROXY` 及对应小写变量。网络参数与其余 uv sync 参数可直接透传，例如：

```bash
UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 bash setup.sh --torch-backend cu130
bash setup.sh --torch-backend cu130 -- --offline
bash setup.sh --torch-backend cu130 -- --check
```

离线同步需要已安装的 uv、匹配的 Python 和完整依赖缓存。`--` 后的参数原样传给 uv，脚本不额外限制覆盖选项。原 `--probe-only`、`--skip-probes` 及各类测速／镜像命令行选项已删除。

## 维护与验证范围

依赖声明变更后运行 `uv lock` 更新锁文件，再执行安装和设备检查。项目为 uv 管理的应用仓库，`package = false`，不构建发行包；IDE 解释器选择 `.venv/bin/python`。

```bash
uv lock --check
uv sync --locked --extra cu130 --check
uv pip check --python .venv/bin/python
```

Python 版本声明、依赖声明、锁文件及本说明应同步维护。
