# PROOF 教师与 SuperMimic

本项目以 [proof-loco@ae910b8](https://github.com/syw-robotics/proof-loco/tree/ae910b821d635f04226489df80bdd77d1d7b0b82) 的教师网络和普通地形预训练奖励为基准，使用当前 SuperMimic 世界模型与可微想象更新。算法位于 `agents/supermimic/`，任务观测、奖励和终止位于 `src/tasks/locomotion/go2_supermimic/`，集中参数位于 `src/config/go2/go2_reference_supermimic_config.py`。

这是高程图教师阶段。当前 `Rough` 任务持续使用原有十类地形、正反金字塔斜坡及梅花桩，不会在固定轮次切换为 parkour。梅花桩占训练环境的10%，每块仍为8×8米，中央桩区为7.5×7.5米，四边各留0.25米平地。六档桩面边长依次为30、27、24、21、18、15厘米；原生网格节距约41.67厘米，实际桩间净间隙依次约为11.67、14.67、17.67、20.67、23.67、26.67厘米。中央出生平台采用原生网格对齐，六档正方形边长依次为2.80、2.77、2.74、2.71、2.68、2.65米，四边出口净间隙与本档桩间净间隙一致。作者原训练采用 PPO，其迭代数不能直接换算为本项目 SuperMimic 的训练预算。

## 网络与输入

| 对象 | 规格 |
|---|---|
| 本体 | 单帧45维：body角速度3、投影重力3、命令3、关节偏差12、关节速度12、上一步动作12；历史10帧旧到新 |
| 本体编码 | `[128,64]→32`，隐藏层LayerNorm与ELU，前三维用真实同一时刻body线速度监督 |
| 高程 | 当前162点，9×18、x内层，1.7×0.8m、间隔0.1m、前移0.6m，值为`base_z-terrain_z-0.25`并裁到±10 |
| 高程编码 | `[64,32]→32`，隐藏层LayerNorm与ELU，输出tanh后乘回合级二值mask |
| 动作 | 本体latent32＋归一化当前帧45＋高程latent32，共109维，经`[512,256,128]→12` |
| critic | 80维干净特权状态＋162点地图＋43维任务记忆，经`[512,256,128]→1` |
| 世界模型 | 346维输入，102维输出：97维物理残差／辅助量＋5维安全事件logit；奖励与失败由任务计算 |

外部本体历史保持逐帧布局；策略内部按观测项重排，与作者编码器输入一致。地图不加观测噪声、不堆叠历史、不存入压缩回放，按当前或记录状态从静态地形重算。训练时，台阶、斜坡和随机格块每次reset有40%概率关闭高程latent；平地与梅花桩始终保持mask=1，其中梅花桩为本项目新增要求。播放及复用播放入口的评估始终启用高程，所有地形保持mask=1。critic与世界模型始终看到干净地图。新增梅花桩改变地图几何，旧地图检查点不能直接用于新地图的严格恢复。

本体历史与critic完整输入采用作者累计总体方差归一化，分母为`std+0.01`；actor的高程与mask不归一化。两个编码器保留默认初始化，动作与价值主干各Linear采用gain=0.01的正交初始化。SuperMimic继续使用确定性想象动作、真实采集固定高斯探索与原优化排期。

Critic的43维任务记忆按main表达接触状态、腾空／支撑计时、前前步动作和命令条件，并补充当前奖励使用的上一腾空／支撑时长、首次接触及回合时间。原前一步动作已在80维特权状态中；heading目标转换为相对机身yaw的sin/cos，步态相位不再使用。该输入扩展是本项目针对SuperMimic的适配，当前总宽度285，需使用匹配任务语义的新Critic权重。

真实状态追加任意足绊脚与四条小腿接触的0/1标签；世界模型用未加类别权重的BCEWithLogits监督对应概率。物理量继续按原97维MSE训练。奖励直接读取事件字段：真实端保持官方硬事件计分，想象端使用概率计算期望惩罚，梯度穿过世界模型进入策略。这是本项目SuperMimic适配，不属于官方PPO中的事件预测机制，也不代表真实碰撞已得到改善。

默认站姿按本项目main：机身初始高度0.29米，四条大腿角均为0.9、膝角−1.8弧度，髋角保持左+0.1／右−0.1。站立偏差采用平方和、权重−2，保留当前站立门控；新增main的pose奖励，权重0.3，按站立／行走／跑步选择关节容差，移动时按body平移与world偏航速度跟踪门控。当前共24项奖励；PD、动作尺度与重置偏移沿用本项目。

按用户要求启用PROOF的`stuck`项，权重−1：决策前平移命令模长严格大于`reward.stuck_velocity_threshold=0.1m/s`、步进后body水平速度模长严格小于同一阈值时返回1，否则为0。纯转向命令不触发；所有地形共用，乘一次控制周期后触发步贡献为−0.02。真实采集和想象使用相同的硬判定；该项没有直接速度梯度，不额外引入平滑代理。

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

`--task-config`加载当前教师配置；`--resume`默认热启动，`--continue-run True`接续迭代号。检查点与同目录`task_config.json`共同描述任务。当前格式为`supermimic_proof_teacher_v2`，状态151维，旧v1教师与MoE检查点均不兼容；不提供字段或权重兼容迁移。本轮事件与站姿版本需重新训练。

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
