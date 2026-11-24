<!-- Copyright (c) 2026-2027 zh -->

# Go2 PROOF 教师任务来源

当前教师以用户指定的 [proof-loco@ae910b821d635f04226489df80bdd77d1d7b0b82](https://github.com/syw-robotics/proof-loco/tree/ae910b821d635f04226489df80bdd77d1d7b0b82) 为基准，适配本项目Go2／MjLab／SuperMimic。论文结构与该后续重实现存在差异，当前采用代码的32+32+45 actor输入，不采用论文64+64结构。

| 本项目位置 | 上游实际位置 | 对齐内容及适配 |
|---|---|---|
| `mdp/reference_task.py`、`state_spec.py` | PROOF `go2_parkour/mdp_cfg/observations_cfg.py`、`config/vanilla_heightmap_parkour_env_cfg.py`，本项目main的任务记忆 | 10帧45维本体、当前162点高程、80维特权状态；Critic另追加43维main式奖励／命令记忆，总285；高程9×18按x内层展开，body到地面高度减0.25；保留作者逐项clip与本体噪声，高程无噪声；本项目关节顺序及足端site不变 |
| `rl/transition_adapter.py`、`mdp/reference_task.py` | `mdp_cfg/events_cfg.py`、`mdp/events.py` | 回合级高程mask，普通地形40%关闭；flat未在作者名单中，因此平地家族始终开启；当前随机格块映射discrete，台阶与斜坡映射对应普通地形 |
| `mdp/reference_rewards.py`与集中reward配置 | `mdp_cfg/rewards_cfg.py`、`mdp/rewards.py`及`use_pretrain_terrain(True)` | 20项基础奖励公式对齐源码，实际权重以集中配置为准。预训练删除stuck、cheat、穿透；时长方差函数实际在非parkour生效，保留该实现，不按名称擅自反转；所有奖励乘一次dt，50Hz失败项为-2/step |
| `foot_slip`、`stand_still_joint_vel_l2` | 本项目`main@9c3a8ad5658dbef1acf4ff98997b64d01e45f90d`的`mdp/reference_rewards.py`及集中reward配置 | 按用户要求追加两项，权重均为−0.1，总计22项；沿用平移命令模长加偏航绝对值的0.1阈值，滑移使用瞬时found与世界水平足速平方，站立使用关节速度平方和。保留PROOF原有feet_slide及其接触历史定义；复用现有状态与想象接触概率，不增加模型维度 |
| 接触与力矩辅助量 | 作者LocoLab [37e567d](https://github.com/syw-robotics/LocoLab/tree/37e567d81ec0198a8984b5c5650cd4489d3a56d3)，`velocity/mdp/{rewards,observations}.py` | 当前足净力先从MuJoCo对地反作用力转换为作用于机器人的世界力，再按正分量>1N后减0.5；滑移使用3物理步历史；非期望接触只计calf。当前资产足端属于calf，按每腿calf几何体净和并排除足几何。限幅前PD力矩从native仿射力模型与实际延迟ctrl恢复，施加力矩另外记录 |
| `mdp/reference_kinematics.py` | 当前Go2 MJCF及原有BlinkMimic FK | 足高按各足yaw方向的四个±0.015m点平均地面；射线起点的+20m不进入高度。想象的last时长／首次落地是接触概率连续代理，真实端读取原生传感器 |
| `mdp/terrains.py` | 原Ri-5036 HIM地形与PROOF `utils/terrains/terrains_cfg.py` | 保留原十类并末尾追加正反光滑金字塔坡，梯度0–0.45、平台2/3m；使用MjLab原生高度场，不复制作者额外roughness。课程保持原六级，整图居中引起原块y坐标平移-8m，局部几何不变 |
| 终止 | PROOF `mdp_cfg/terminations_cfg.py` | 倾角>1.5rad、base历史净力>5N为失败；20s超时与核心边界外4m为truncation。想象倾角软失败保留现有SuperMimic，base接触使用来源硬阈值 |

Go2资产、执行器PD、动作尺度、物理随机化、命令采样、行走表现课程及SuperMimic训练预算沿用本项目。参考实现为IsaacLab3／PhysX，本项目为MjLab1.6／MuJoCo；不将代码对拍或短时训练当作作者性能复现或真机验证。原资产与FK来源说明、`LICENSE_blinkmimic`保持原归属。

算法、加速与第三方许可见 [算法来源说明](../../../../agents/supermimic/NOTICE.md)，配置、运行与三输入导出接口见 [算法与任务说明](../../../../agents/README.md)。
