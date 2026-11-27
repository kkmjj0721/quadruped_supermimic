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

`setup.sh` 定位自身所在仓库，通过 `tools/setup_env.py` 和 `tools/setup_downloads.py` 测速并选择下载源，调用 uv 安装 `.python-version` 指定的 Python 和项目依赖。已有环境与锁文件一致时直接结束。脚本不安装系统驱动或 CUDA Toolkit。

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

## CUDA 编译数学库路径

训练启用 `torch.compile` 时，PyTorch 会查找 CUDA Toolkit 的 `libdevice.10.bc`。机器同时安装多套 Toolkit，且 PATH 首先找到 `/usr/bin/nvcc` 时，自动推导的 CUDA 根目录可能是 `/usr`，从而出现查找 `/usr/nvvm/libdevice/libdevice.10.bc` 失败的警告。

CUDA 训练入口会在创建 runner 前设置 `TRITON_LIBDEVICE_PATH`，默认使用 `/usr/local/cuda/nvvm/libdevice/libdevice.10.bc` 并解析符号链接，避免依赖 PATH 中的 `nvcc`。已有显式设置时保持原值。编译训练需要该数学库；路径无效时由编译器直接报错，不静默改用其他数学库。

Toolkit 不在默认目录或需要固定版本时，在启动训练的 shell 中显式指定路径。例如 Toolkit 安装在 `/usr/local/cuda-13.0` 时：

```bash
export TRITON_LIBDEVICE_PATH=/usr/local/cuda-13.0/nvvm/libdevice/libdevice.10.bc
```

需要在后续交互式 SSH 会话中固定使用该版本时，将该行加入 `~/.bashrc`，然后重新登录；已有终端需先执行该行。非交互式作业应在作业启动脚本中设置该变量。已启动的训练进程不会获得后续的环境变量修改。

## 多源测速与慢网安装

默认分别比较三类下载通道，避免普通包索引很快、PyTorch 大包仍然很慢：

| 类别 | 实际测速样本 | 内置候选 |
| --- | --- | --- |
| Python | uv 返回的当前平台固定版本 archive | Astral、GitHub、南大、USTC |
| 普通 PyPI | 锁定的 numpy CPython 3.11 Linux wheel | 官方、清华、阿里、华为 |
| PyTorch | 锁定后端的 torch 和 torchvision 两个 wheel | 官方 R2、官方、阿里、南大 |

每个样本最多读取 512 KiB，按实际前缀吞吐排名。每源使用单个测速子进程，源的总墙钟时限为 `--probe-timeout` 乘样本数量，包含连接和响应头等待；速度统计不包含解释器启动时间。同源任一样本失败或为空即不可用。前缀测速不能保证完整大文件始终保持同样速度，也不提前审计文件类型；实际安装由 uv 按锁定校验信息核验文件。

正常安装先采用排名靠前的源，安装失败或超时后依次尝试其他 PyPI／Torch 组合，复用 uv 已下载的缓存。Python 尚未安装时也按测速排名换源安装；已有解释器直接复用。全部候选失败时返回失败结果，不静默改变依赖版本或计算后端。

只查看测速结果：

```bash
bash setup.sh --torch-backend cu130 --probe-only
```

只测速不会安装项目依赖；要求固定版本的 uv Python 已存在，助手通过它读取锁文件。安装时系统 `python3` 启动助手，准备好固定版本后切换到对应解释器。无需先安装项目的重型依赖。

服务器只能访问部分镜像时，可重复指定各类别的根地址；该类别只使用显式给出的列表。例如：

```bash
bash setup.sh --torch-backend cu130 \
  --python-mirror https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone \
  --pypi-mirror https://repo.huaweicloud.com/repository/pypi/packages \
  --pypi-mirror https://mirrors.aliyun.com/pypi/packages \
  --torch-mirror https://mirrors.aliyun.com/pytorch-wheels \
  --torch-mirror https://mirror.nju.edu.cn/pytorch/whl
```

`--pypi-mirror` 包含 `/packages`，后面保留原锁文件的 artifact 路径；`--torch-mirror` 后面追加 `/cpu/<文件名>` 或 `/cu130/<文件名>`。Python 未传命令行镜像时，优先使用 `UV_PYTHON_INSTALL_MIRROR`，再使用内置列表。镜像需要提供锁定版本的相同文件，候选地址不保证在每个服务器或每个时刻都可访问。

调整测速和每轮实际安装预算：

```bash
bash setup.sh --torch-backend cu130 \
  --probe-timeout 5 --probe-bytes 1048576 --download-timeout 600
```

原项目先执行 `uv lock --check`。选择镜像后，助手直接写入持久下载项目 `.artifacts/setup-downloads/<backend>/`：下载锁仅替换 artifact URL，依赖版本、registry、markers 与上游文件校验信息沿用原锁。随后 `uv sync --frozen` 安装到仓库 `.venv`。原仓库的 `uv.lock` 和 `pyproject.toml` 不被改写，下载项目保留最后一次选择，便于查看；不创建临时替换目录或额外计算锁文件哈希。

## 原始下载通道、代理与离线模式

需要完全由 uv 处理索引、项目或本地路径参数时，显式选择原始锁定通道；例如自定义解析索引时先更新锁文件，再安装：

```bash
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv lock
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  bash setup.sh --torch-backend cu130 --skip-probes
```

`--skip-probes` 使用原项目的 `uv sync --locked`，不做镜像测速或 artifact URL 改写。索引配置方式见 [uv 配置说明](https://docs.astral.sh/uv/concepts/configuration-files/)，锁定与冻结同步语义见 [uv 锁定与同步](https://docs.astral.sh/uv/concepts/projects/sync/)。

代理使用标准 `HTTP_PROXY`、`HTTPS_PROXY` 及对应小写变量。其余 uv sync 参数放在 `--` 后，原样透传，不增加参数黑名单：

```bash
UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 bash setup.sh --torch-backend cu130
bash setup.sh --torch-backend cu130 -- --offline
bash setup.sh --torch-backend cu130 -- --check
bash setup.sh --torch-backend cu130 -- --dry-run
```

`--offline`（或 `UV_OFFLINE=1`）、`--check`、`--dry-run` 使用原始锁定通道并跳过测速。离线同步需要已有 uv、匹配的 Python 和依赖缓存；离线加 `--probe-only` 仅报告未运行网络测速，也不安装任何内容。

## 维护与验证范围

依赖声明变更后运行 `uv lock` 更新锁文件，再执行安装和设备检查。项目为 uv 管理的应用仓库，`package = false`，不构建发行包；IDE 解释器选择 `.venv/bin/python`。

```bash
uv lock --check
uv sync --locked --extra cu130 --check
uv pip check --python .venv/bin/python
```

Python 版本声明、依赖声明、锁文件及本说明应同步维护。
