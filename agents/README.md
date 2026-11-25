# PROOF 教师与 SuperMimic

本项目以 [proof-loco@ae910b8](https://github.com/syw-robotics/proof-loco/tree/ae910b821d635f04226489df80bdd77d1d7b0b82) 的教师网络和普通地形预训练奖励为基准，使用当前 SuperMimic 世界模型与可微想象更新。算法位于 `agents/supermimic/`，任务观测、奖励和终止位于 `src/tasks/locomotion/go2_supermimic/`，集中参数位于 `src/config/go2/go2_reference_supermimic_config.py`。

这是高程图教师阶段。当前 `Rough` 任务持续使用原有十类地形与正反金字塔斜坡，共十二类，不会在固定轮次切换为 parkour。作者原训练采用 PPO，其迭代数不能直接换算为本项目 SuperMimic 的训练预算。

## 网络与输入

| 对象 | 规格 |
|---|---|
| 本体 | 单帧45维：body角速度3、投影重力3、命令3、关节偏差12、关节速度12、上一步动作12；历史10帧旧到新 |
| 本体编码 | `[128,64]→32`，隐藏层LayerNorm与ELU，前三维用真实同一时刻body线速度监督 |
| 高程 | 当前162点，9×18、x内层，1.7×0.8m、间隔0.1m、前移0.6m，值为`base_z-terrain_z-0.25`并裁到±10 |
| 高程编码 | `[64,32]→32`，隐藏层LayerNorm与ELU，输出tanh后乘回合级二值mask |
| 动作 | 本体latent32＋归一化当前帧45＋高程latent32，共109维，经`[512,256,128]→12` |
| critic | 80维干净特权状态＋162点地图＋43维任务记忆，经`[512,256,128]→1` |
| 世界模型 | 293维输入含92维身体及足下地形，119维输出含97维连续量与22维接触事件 logits |

外部本体历史保持逐帧布局；策略内部按观测项重排，与作者编码器输入一致。地图不加观测噪声、不堆叠历史、不存入压缩回放，按当前状态从静态地形重算。训练时，台阶、斜坡和随机格块每次reset有40%概率关闭高程latent；平地始终保持mask=1。播放及复用播放入口的评估始终启用高程，所有地形保持mask=1。critic与世界模型始终看到干净地图。地形集合变化会改变地图几何，检查点严格恢复要求与原训练地图一致。

本体历史与critic完整输入采用作者累计总体方差归一化，分母为`std+0.01`；actor的高程与mask不归一化。两个编码器保留默认初始化，动作与价值主干各Linear采用gain=0.01的正交初始化。SuperMimic继续使用确定性想象动作、真实采集固定高斯探索与原优化排期。

Critic的43维任务记忆按main表达接触状态、腾空／支撑计时、前前步动作和命令条件，并补充当前奖励使用的上一腾空／支撑时长、首次接触及回合时间。原前一步动作已在80维特权状态中；heading目标转换为相对机身yaw的sin/cos，步态相位不再使用。该输入扩展是本项目针对SuperMimic的适配，当前总宽度285，需使用匹配任务语义的新Critic权重。

训练保留失败和超时动作对应的真实终态，从下一步起屏蔽跨回合监督。足部瞬时接触按世界合力模长超过 1 N 标记，足部历史接触覆盖一个控制周期。两组四足标签分别采用权重 0.5 的 BCE，绊脚与13部位的非期望接触共用权重 1 的事件 BCE，连续量按标准化 MSE 监督。真实步态奖励使用 MjLab 足端传感器按物理子步记录的腾空、支撑时长和本控制步的首次触地标记。想象步用预测接触概率递推这些量。

世界模型在每步预测的位姿上查询机身网格和球足下的地形。真实失败根据机身倾角超过1.5 rad或碰撞外形净空为负判定；机身接触力参与非期望接触奖励，不单独触发终止。想象用倾角与几何净空给出生存概率。任一足的“前一步离地概率 × 本步接触概率 × 足底缺乏支撑比例”超过 0.3 时，策略保留当步奖励，后续用落脚后状态的 critic 估值。新任务规格需要重新训练世界模型，旧训练检查点无法严格续训。

默认站姿按本项目main：机身初始高度0.29米，四条大腿角均为0.9、膝角−1.8弧度，髋角保持左+0.1／右−0.1。普通地形的站立偏差采用平方和、权重−2，保留当前站立门控；main的pose奖励权重0.3，按站立／行走／跑步选择关节容差，移动时按body平移与world偏航速度跟踪门控。普通地形共25项奖励；PD、动作尺度与重置偏移沿用本项目。

两阶段的`undesired_contacts`均按四腿hip／thigh／calf及base共13部位计数，足端几何排除。同一部位的多个碰撞体先求净合力，取最近4个物理子步的最大模长，严格超过1 N时该部位记1；接触对象不作过滤，最多计13，权重−1。base包含机身箱体和头部碰撞体，同一部位仍只计一次。想象使用对应事件概率求和。`body_orientation_l2`沿用main的投影重力xy平方和，两个阶段权重均为−0.1。

按用户要求启用PROOF的`stuck`项，权重−2：决策前平移命令模长严格大于`reward.stuck_velocity_threshold=0.1m/s`、步进后body水平速度模长严格小于同一阈值时返回1，否则为0。纯转向命令不触发；所有地形共用，乘一次默认控制周期后触发步贡献为−0.04。真实采集和想象使用相同的硬判定；该项没有直接速度梯度，不额外引入平滑代理。

## 训练、续训与播放

从仓库根目录运行：

```bash
uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Rough \
  --device cuda:0 --num-envs 64 --max-iterations 1000
uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Rough \
  --checkpoint-file '<实验目录>/model_1000.pt' --num-envs 1
```

训练保留编译、CUDA Graph、BF16、Muon＋Adam、想象／critic微批、frames压缩回放和设备指标汇总。相关CLI参数仍为`--wm-compile`、`--cuda-graph`、`--wm-bf16`、`--actor-bf16`、`--critic-bf16`、`--imag-microbatch`、`--critic-microbatch`、`--replay-history-storage`与`--metric-interval`。CPU执行时显式关闭三个BF16开关。完整参数通过入口`--help`查看。

启动耗时与训练耗时共同计入执行成本。世界模型复用单步监督编译核，想象复用单步编译核，避免展开整个时间窗口的大图编译；Muon的原张量更新由外层CUDA图执行，不再单独进行Inductor编译。

CUDA执行将整轮世界模型采样、更新与指标累计合入一张图，种子采样、actor和critic更新按策略冻结状态合入另一张图。详细计时对应融合阶段`wm_sampling_update`与`policy_sampling_update`，原世界模型／控制总时间仍保留；指标默认每10轮回读，护栏和调用结束时也会完成当前窗口。

`--task-config`加载当前教师配置；`--resume`默认热启动，`--continue-run True`接续迭代号。检查点与同目录`task_config.json`共同描述任务。当前格式为`supermimic_proof_teacher_v2`，状态168维；151维旧状态、旧v1教师与MoE检查点均无法严格续训，不提供字段或权重兼容迁移。扩展接触范围后需重新训练世界模型。

任务名称沿用`Unitree-Go2-SuperMimic-Flat/Rough`。当前Go2资产、关节顺序、PD、动作尺度、物理随机化、命令采样和地形课程沿用本项目；因此不声称与作者IsaacLab／PhysX轨迹等价。

## 二阶段 Parkour

任务名为 `Unitree-Go2-SuperMimic-Parkour`。二阶段复用本项目 Go2、PD、随机化和 SuperMimic 训练器，采用 [proof-loco ae910b8 的 Parkour 地形](https://github.com/syw-robotics/proof-loco/blob/ae910b8/source/proof_parkour/proof_parkour/utils/terrains/terrains_cfg.py)。地图含 10 行难度、20 个实际地形列和 11 个地形族，每块 8×8 米。其中沟 6 列，高台 4 列；其他列为跨栏、高落差下阶、直楼梯、金字塔楼梯、正反坡和离散障碍。随仓库提供种子 17 的预生成地图。坡面与离散障碍从作者三角网格以 2 cm 间距采样。生成其他种子的方法见 [地形资产生成工具](../tools/build_proof_parkour_terrain.py)。

没有检查点时可以使用零动作查看器浏览地形。正式训练使用本项目的一阶段 Rough 权重及同目录的 `task_config.json`。二阶段读入来源网络配置，并使用二阶段的命令、重置、地形课程与 24 项奖励。热启动沿用形状匹配的网络和优化器，在新地图上重新收集回放；`--continue-run True` 用于同一任务规格的续训。默认二阶段产物目录为 `logs/supermimic/go2_supermimic/parkour/<运行标识>/`。

```bash
uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Parkour --agent zero --num-envs 1

uv run --no-sync python scripts/train.py Unitree-Go2-SuperMimic-Parkour \
  --resume '<一阶段 Rough 目录>/model_10000.pt' --device cuda:0 --num-envs 64

uv run --no-sync python scripts/play.py Unitree-Go2-SuperMimic-Parkour \
  --checkpoint-file '<二阶段目录>/model_10000.pt' --num-envs 1
```

二阶段 actor 继续使用 450 维本体历史、162 点高程和回合级遮蔽值。训练中跨栏、沟、高台和高落差下阶始终开启高程；六类直行地形使用前进命令。穿透奖励仅针对沟、高台和高落差下阶的虚拟边缘。二阶段回放上下文额外记录回合所属地形族，真实轨迹和想象轨迹均按这项信息计算奖励。

## 策略导出

保存检查点时生成三输入`policy.onnx`。显式离线导出：

```python
from src.tasks.locomotion.go2_supermimic.rl.checkpoints import export_checkpoint
export_checkpoint("<实验目录>/model_1000.pt", "<导出目录>")
```

TorchScript与ONNX均接收`obs_history[B,450]`、`heightmap[B,162]`和`exteroceptive_mask[B,1]`，输出未缩放动作`actions[B,12]`；关节目标为`default_joint_positions+actions*action_scales`。导出为FP32，并保存原训练精度、历史布局、地图及控制坐标等元数据，格式为`supermimic_proof_teacher_policy_v1`。播放和评估使用同一套输入构造。

## 来源与许可

算法与加速来源见 [SuperMimic来源说明](supermimic/NOTICE.md)，任务、观测和奖励的对应关系见 [Go2任务来源说明](../src/tasks/locomotion/go2_supermimic/NOTICE.md)。短时接口检查不代表策略收敛、论文性能或真机部署已验证。
