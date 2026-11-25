<!-- Copyright (c) 2026-2027 zh -->

# Go2 PROOF 教师任务来源

当前教师以用户指定的 [proof-loco@ae910b821d635f04226489df80bdd77d1d7b0b82](https://github.com/syw-robotics/proof-loco/tree/ae910b821d635f04226489df80bdd77d1d7b0b82) 为基准，适配本项目Go2／MjLab／SuperMimic。论文结构与该后续重实现存在差异，当前采用代码的32+32+45 actor输入，不采用论文64+64结构。

| 本项目位置 | 上游实际位置 | 对齐内容及适配 |
|---|---|---|
| `mdp/reference_task.py`、`state_spec.py` | PROOF `go2_parkour/mdp_cfg/observations_cfg.py`、`config/vanilla_heightmap_parkour_env_cfg.py`，本项目main的任务记忆 | 10帧45维本体、当前162点高程、80维特权状态；Critic另追加43维main式奖励／命令记忆，总285；高程9×18按x内层展开，body到地面高度减0.25；保留作者逐项clip与本体噪声，高程无噪声；本项目关节顺序及足端site不变 |
| `rl/transition_adapter.py`、`mdp/reference_task.py` | `mdp_cfg/events_cfg.py`、`mdp/events.py` | 训练采用回合级高程mask，台阶、斜坡和随机格块40%关闭；flat未在作者名单中，因此平地家族始终开启；梅花桩按用户要求始终开启；本项目播放始终开启高程；当前随机格块映射discrete，台阶与斜坡映射对应普通地形 |
| `mdp/reference_rewards.py`与集中reward配置 | `mdp_cfg/rewards_cfg.py`、`mdp/rewards.py`及`use_pretrain_terrain(True)` | 基础奖励以PROOF源码为准；按用户要求，stand_still采用main的关节偏差平方和与−2权重，保留PROOF分量门控，其余实际权重以集中配置为准。绊脚与calf接触在真实端生成0/1标签、想象端用监督事件概率计算期望惩罚。上游预训练删除stuck、cheat、穿透，本项目按用户要求重新启用stuck；时长方差函数实际在非parkour生效，保留该实现，不按名称擅自反转；所有奖励乘一次dt，50Hz失败项为-2/step |
| `stuck` | PROOF `mdp_cfg/rewards_cfg.py:144`及LocoLab `velocity/mdp/rewards.py:604`的`stuck_penalty` | 按用户要求追加，权重−1；决策前平移命令模长>0.1m/s且步进后body水平速度模长<0.1m/s时原值为1，其他情况为0；所有地形启用，阈值独立配置，dt=.02时触发步贡献−.02。沿用来源硬判定，想象中没有直接速度梯度，不增加模型维度 |
| `foot_slip`、`stand_still_joint_vel_l2` | 本项目`main@9c3a8ad5658dbef1acf4ff98997b64d01e45f90d`的`mdp/reference_rewards.py`及集中reward配置 | 按用户要求追加两项，权重均为−0.1；沿用平移命令模长加偏航绝对值的0.1阈值，滑移使用瞬时足力模长大于1 N的接触标记与世界水平足速平方，站立使用关节速度平方和。保留PROOF原有feet_slide及其接触历史定义；复用现有状态与想象接触概率，不增加模型维度 |
| `pose` | 本项目`main@9c3a8ad`的`mdp/gait_rewards.py::posture_reward`与`mdp/reference_rewards.py`跟踪门控 | 按用户要求恢复完整分段姿态分与速度跟踪门控，权重调整为0.3；与stuck合计24项奖励。容差与阈值直接采用main；门控保留body平移与world偏航坐标，PROOF其余速度奖励仍使用body偏航。仅迁入pose，不恢复main步态奖励 |
| 接触与力矩辅助量 | 作者LocoLab [37e567d](https://github.com/syw-robotics/LocoLab/tree/37e567d81ec0198a8984b5c5650cd4489d3a56d3)，`velocity/mdp/{rewards,observations}.py` | 当前足净力先从MuJoCo对地反作用力转换为作用于机器人的世界力，再按模长>1 N标记瞬时触地。critic 使用接触值减0.5，历史滑移使用整个控制周期；非期望接触只计calf。当前资产足端属于calf，按每腿calf几何体净和并排除足几何。限幅前PD力矩从native仿射力模型与实际延迟ctrl恢复，施加力矩另外记录 |
| 事件概率监督 | 本项目SuperMimic适配，官方PPO没有对应预测头 | 物理视图147维＋四元数4维，模型输入276／输出102。足端瞬时与历史接触、绊脚和小腿接触按组做 BCE，89个连续量用标准化 MSE。真实与想象奖励统一读取事件字段；不把概率校准或能力改善视为已验证 |
| 默认站姿 | 本项目`main@9c3a8ad`的集中`init_state`配置 | 按用户要求恢复base z=.29、四条thigh=.9、calf=-1.8、hip左右±.1；当前MuJoCo资产前向计算四足球底净空均约.01249m。PD及动作尺度不变 |
| `mdp/reference_kinematics.py` | 当前Go2 MJCF及原有BlinkMimic FK | 足高按各足yaw方向的四个±0.015m点平均地面；射线起点的+20m不进入高度。接触时钟使用同一控制步递推，真实端以足力阈值标记，想象端传入概率 |
| `mdp/terrains.py`、`mdp/stepping_stones.py` | 原Ri-5036 HIM地形、PROOF `utils/terrains/terrains_cfg.py`及MjLab 1.6.0 `BoxSteppingStonesTerrainCfg`的箱体／边框／平台思路 | 保留原十二类，末尾梅花桩占10%环境；8×8m地块包含7.5×7.5m桩区及各0.25m平地。按用户要求独立插值六档桩宽35→25cm、实际净间隙10→15cm，节距45→40cm；以地块中心对称铺桩，边界箱体裁至桩区。平台最小边长2.5m、按网格等缝对齐，实际边长3.05/2.97/2.89/2.81/2.73/2.65m，出口缝与本档桩间缝相同。保留桩顶0.1m、难度递增±2cm高差及坑底−0.5m；独立间隙课程是本项目适配，不声称为上游原生参数行为 |
| 终止 | PROOF `mdp_cfg/terminations_cfg.py` | 真实端根据倾角、机身历史净力或碰撞外形净空终止，超时与越界记为截断。想象使用 TGI 的几何生存概率与落脚回报，见算法来源说明 |

Go2资产、执行器PD、动作尺度、物理随机化、命令采样、行走表现课程及SuperMimic训练预算沿用本项目。参考实现为IsaacLab3／PhysX，本项目为MjLab1.6／MuJoCo；不将代码对拍或短时训练当作作者性能复现或真机验证。原资产与FK来源说明、`LICENSE_blinkmimic`保持原归属。

算法、加速与第三方许可见 [算法来源说明](../../../../agents/supermimic/NOTICE.md)，配置、运行与三输入导出接口见 [算法与任务说明](../../../../agents/README.md)。
