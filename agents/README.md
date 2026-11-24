# PROOF 教师与 SuperMimic

本项目以 [proof-loco@ae910b8](https://github.com/syw-robotics/proof-loco/tree/ae910b821d635f04226489df80bdd77d1d7b0b82) 的教师网络和普通地形预训练奖励为基准，使用当前 SuperMimic 世界模型与可微想象更新。算法位于 `agents/supermimic/`，任务观测、奖励和终止位于 `src/tasks/locomotion/go2_supermimic/`，集中参数位于 `src/config/go2/go2_reference_supermimic_config.py`。

这是高程图教师阶段。当前 `Rough` 任务持续使用原有十类地形与新增正反金字塔斜坡，不会在固定轮次切换为 parkour。作者原训练采用 PPO，其迭代数不能直接换算为本项目 SuperMimic 的训练预算。

## 网络与输入

| 对象 | 规格 |
|---|---|
| 本体 | 单帧45维：body角速度3、投影重力3、命令3、关节偏差12、关节速度12、上一步动作12；历史10帧旧到新 |
| 本体编码 | `[128,64]→32`，隐藏层LayerNorm与ELU，前三维用真实同一时刻body线速度监督 |
| 高程 | 当前162点，9×18、x内层，1.7×0.8m、间隔0.1m、前移0.6m，值为`base_z-terrain_z-0.25`并裁到±10 |
| 高程编码 | `[64,32]→32`，隐藏层LayerNorm与ELU，输出tanh后乘回合级二值mask |
| 动作 | 本体latent32＋归一化当前帧45＋高程latent32，共109维，经`[512,256,128]→12` |
| critic | 80维干净特权状态＋162点地图＋43维任务记忆，经`[512,256,128]→1` |
| 世界模型 | 341维输入，97维残差／辅助输出；奖励与失败由任务解析计算 |

外部本体历史保持逐帧布局；策略内部按观测项重排，与作者编码器输入一致。地图不加观测噪声、不堆叠历史、不存入压缩回放，按当前或记录状态从静态地形重算。普通粗糙地形每次reset有40%概率关闭高程latent；作者名单没有flat，平地家族保持mask=1。critic与世界模型始终看到干净地图。

本体历史与critic完整输入采用作者累计总体方差归一化，分母为`std+0.01`；actor的高程与mask不归一化。两个编码器保留默认初始化，动作与价值主干各Linear采用gain=0.01的正交初始化。SuperMimic继续使用确定性想象动作、真实采集固定高斯探索与原优化排期。

Critic的43维任务记忆按main表达接触状态、腾空／支撑计时、前前步动作和命令条件，并补充当前奖励使用的上一腾空／支撑时长、首次接触及回合时间。原前一步动作已在80维特权状态中；heading目标转换为相对机身yaw的sin/cos，步态相位不再使用。该输入扩展是本项目针对SuperMimic的适配，当前总宽度285，需使用匹配任务语义的新Critic权重。

## 训练、续训与播放

从仓库根目录运行：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Rough \
  --device cuda:0 --num-envs 64 --max-iterations 1000
uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Rough \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1
```

训练保留编译、CUDA Graph、BF16、Muon＋Adam、想象／critic微批、frames压缩回放和设备指标汇总。相关CLI参数仍为`--wm-compile`、`--cuda-graph`、`--wm-bf16`、`--actor-bf16`、`--critic-bf16`、`--imag-microbatch`、`--critic-microbatch`、`--replay-history-storage`与`--metric-interval`。CPU执行时显式关闭三个BF16开关。完整参数通过入口`--help`查看。

`--task-config`加载当前教师配置；`--resume`默认热启动，`--continue-run True`接续迭代号。检查点与同目录`task_config.json`共同描述任务。新格式为`supermimic_proof_teacher_v1`，旧MoE检查点不能加载为高程图教师；不提供字段或权重兼容迁移。

任务名称沿用`Unitree-Go2-SuperMimic-Flat/Rough`。当前Go2资产、关节顺序、PD、动作尺度、物理随机化、命令采样和地形课程沿用本项目；因此不声称与作者IsaacLab／PhysX轨迹等价。

## 策略导出

保存检查点时生成三输入`policy.onnx`。显式离线导出：

```python
from src.tasks.locomotion.go2_supermimic.rl.checkpoints import export_checkpoint
export_checkpoint("<实验目录>/model_1000.pt", "<导出目录>")
```

TorchScript与ONNX均接收`obs_history[B,450]`、`heightmap[B,162]`和`exteroceptive_mask[B,1]`，输出未缩放动作`actions[B,12]`；关节目标为`default_joint_positions+actions*action_scales`。导出为FP32，并保存原训练精度、历史布局、地图及控制坐标等元数据，格式为`supermimic_proof_teacher_policy_v1`。播放和评估使用同一套输入构造。

## 来源与许可

算法与加速来源见 [SuperMimic来源说明](supermimic/NOTICE.md)，任务、观测和奖励的对应关系见 [Go2任务来源说明](../src/tasks/locomotion/go2_supermimic/NOTICE.md)。短时接口检查不代表策略收敛、论文性能或真机部署已验证。
