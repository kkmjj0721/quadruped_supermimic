# 🐾 Quadruped Lab

基于 **MjLab** 的四足机器人运动控制项目，面向 **Unitree Go2**，支持平地、粗糙地形与 Parkour 的速度跟踪、策略训练、可视化播放和策略导出。

以 `proof-loco@ae910b8` 的高程图教师为基准，使用 **SuperMimic 可微世界模型**训练，提供普通地形预训练与 Parkour 二阶段任务。

| 阶段 | 任务标识 | 地形 |
| --- | --- | --- |
| 普通地形 | `Unitree-Go2-SuperMimic-Flat` | 平地 |
| 普通地形 | `Unitree-Go2-SuperMimic-Rough` | 粗糙地形、阶梯与斜坡 |
| 二阶段 | `Unitree-Go2-SuperMimic-Parkour` | 沟壑、攀爬、跨栏、高落差等十族地形 |

策略使用 10 帧本体历史与当前 162 点高程图；训练支持 `torch.compile`、CUDA Graph、BF16、Muon＋Adam 和压缩回放。普通地形与 Parkour 各启用 23 项奖励，具体配方、网络输入和来源差异见 [算法与任务说明](agents/README.md)及[任务来源说明](src/tasks/locomotion/go2_supermimic/NOTICE.md)。

## 🚀 快速开始

环境要求：Linux x86_64、Python 3.11；GPU 训练使用兼容 CUDA 13.0 的 NVIDIA 驱动。以下命令均在仓库根目录执行。

```bash
bash setup.sh --torch-backend cu130
export PATH="$HOME/.local/bin:$PATH"
uv run --no-sync python -B tools/check_install.py --device cuda
```

安装脚本会准备 uv、Python 和项目虚拟环境。CPU 安装与更多选项见 [安装指南](docs/setup.md)；日常运行保留 `--no-sync`，沿用已安装的后端。

## 🧠 训练

```bash
# PROOF 教师：普通粗糙地形预训练
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Rough \
  --device cuda:0 --num-envs 4096 --max-iterations 10000
```

平地使用表中对应的 `Flat` 任务。当前 `Rough` 包含原有地形及正反金字塔斜坡，保持在普通地形阶段。环境数与训练轮数按资源和实验目标调整；至少使用 2 个并行环境，网络、续训与导出接口见 [算法与任务说明](agents/README.md)。

普通地形训练后，通过检查点热启动 Parkour 二阶段：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Parkour \
  --resume '<Rough实验目录>/model_10000.pt' \
  --device cuda:0 --num-envs 4096 --max-iterations 10000
```

`--resume` 默认加载参数并从第 1 轮训练；同阶段接续轮次时添加 `--continue-run True`，`--max-iterations` 表示目标总轮数。Rough 转 Parkour 使用默认热启动方式，入口会切换二阶段奖励、按地形族命令、重置和域随机化配置。

Parkour 默认读取仓库内的 `src/assets/terrains/proof_parkour_17.npz`：10 行 × 20 列，已移除离散障碍，运行时无需 IsaacLab。调整地形参数或重新生成资产的方法见 [二阶段 Parkour](agents/README.md#二阶段-parkour)。

默认产物目录：

- **Flat／Rough**：`logs/supermimic/go2_supermimic/<运行标识>/`
- **Parkour**：`logs/supermimic/go2_supermimic/parkour/<运行标识>/`

## 🎮 策略播放

将检查点路径替换为实际文件，任务标识决定播放地形；也可将 Rough 权重放到 Parkour 任务中查看：

```bash
# 播放 SuperMimic 策略
uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Rough \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1

# 播放 Parkour 策略
uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Parkour \
  --checkpoint-file '<Parkour实验目录>/model_10000.pt' --num-envs 20
```

查看器默认自动选择 MuJoCo 窗口或 Viser 浏览器界面，也可通过 `--viewer native` / `--viewer viser` 指定。添加 `--video True` 可将录像保存到实验目录的 `videos/play/`。

没有检查点时，可用 `--agent zero --num-envs 1` 查看环境。SuperMimic 播放需要保留检查点同目录的 `task_config.json`。完整参数可在训练或播放命令后添加 `--help` 查看。

## 📦 策略导出

保留检查点同目录的 `task_config.json`，显式导出 TorchScript、ONNX 与部署元数据：

```bash
uv run --no-sync python scripts/export.py '<实验目录>/model_10000.pt' '<导出目录>'
```

导出策略接收本体历史、高程图和高程遮蔽三个输入，输出 12 维未缩放动作；输入布局及关节目标转换见 [策略导出](agents/README.md#策略导出)。

## 项目结构

```text
src/tasks/   # 任务环境、奖励与课程
src/config/  # 机器人与任务配置
src/assets/  # 机器人模型与地形资产
agents/      # SuperMimic 算法
scripts/     # 训练、播放与导出入口
tools/       # 环境安装、检查与离线地形构建
docs/        # 安装与任务用法
```

## 文档与许可

- [算法与任务说明](agents/README.md)：配置、续训、高程输入与导出。
- [Go2 配置](src/config/go2/)：机器人、控制与任务参数。

项目采用 [MIT 许可](LICENSE)；引入的算法与机器人资产遵循各自目录中的许可及来源说明。
