<!-- Copyright (c) 2026-2027 zh -->

# SuperMimic 与 PROOF 教师来源

本目录由本项目维护，不依赖参考仓库的运行环境。当前任务采用PROOF官方教师网络与SuperMimic可微世界模型，任务特定观测、奖励、终止及地形规则留在`src/tasks/locomotion/go2_supermimic/`。配置与运行方式见 [算法与任务说明](../README.md)。

| 来源 | 固定版本与位置 | 采用内容及差异 |
|---|---|---|
| PROOF官方教师 | [proof-loco@ae910b8](https://github.com/syw-robotics/proof-loco/tree/ae910b821d635f04226489df80bdd77d1d7b0b82)，`source/z_rl_proof/proof_rl/models/proof_masked_teacher_model.py`、教师`z_rl_proof_ppo_cfg.py` | 本体与高程32维双编码、LayerNorm／ELU、高程tanh与回合mask、归一化当前帧直连、速度MSE；不迁入PPO、深度学生、不确定性估计或蒸馏 |
| 作者Z_RL依赖 | [z_rl@4c8f0a2](https://github.com/syw-robotics/z_rl/tree/4c8f0a2d1b7e7366d34edcdac5d060c53613451c)，`models/variants/group_mlp_encoder_model.py`、`modules/{mlp,normalization}.py`与`algorithms/ppo.py` | 隐藏层LN位置、累计总体方差/std+0.01、零初始计数，以及仅actor/critic主干的正交初始化；PROOF未锁依赖，本项目记录实际核对版本，不直接加载作者权重 |
| SuperMimic参考实现 | [go2-supermimic@d9eee9b](https://github.com/Zhengtianjin/go2-supermimic/tree/d9eee9b3f248e5300e636c433255db2c1c3e086a)，`supermimic/{train,world_model,buffer,imagine}.py` | 保留自回归物理监督、短时域BPTT、末端价值和lambda-return、真实回放与归档；网络和任务奖励改为当前教师 |
| 残差动力学与FK | [BlinkMimic@d264658](https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409)，`train.py`、`networks.py`与`kinematics.py` | 既有残差／物理增量双统计、核心误差下限、零初始化世界模型输出、四元数积分及按本项目资产定义的FK；当前继续使用 |
| Muon＋Adam | [PartyMimic@69c53621](https://github.com/Roboparty/PartyMimic/tree/69c53621e6394f3c5dd7a923d0a93dcfcd1fb8eb)，`optim.py` | 保留当前矩阵分组、设备学习率、捕获方式和有限性跳步；actor所有Linear权重用Muon，偏置与LayerNorm参数用Adam，WM仅中间Linear用Muon，critic用Adam |
| CUDA Graph及编译 | 项目既有`cuda_graph.py`、`compilation.py`，工程来源[BlinkMimic](https://github.com/fuyichen0725/blinkmimic/tree/d264658d6c9ce61fb52f9f0f8985a369f3ccc409)与[InstantTrack@b9512a9](https://github.com/Huy-Quang-Dao/instantTrack/tree/b9512a96d4e82e6ddea9a6de3b03276888bbcb2c) | 保留更新图、设备采样、整体监督／critic／返回编译、BF16、微批及frames回放；捕获或编译异常直接暴露，不静默降级 |

`policy.py`仅保留`ProofTeacherActor`，不保留旧MoE路径。外部历史是逐帧旧到新，内部显式重排为作者逐观测项历史；高程只提供当前162点。`encode`的前三维速度由真实回放同一时刻标签监督，梯度与想象目标共同更新本体编码器；不使用想象速度作为监督标签。

地形接触、终止帧监督与落脚回报依据 [TGI a64be1f](https://github.com/leggedrobotics/terrain-grounded-imagination/tree/a64be1f185110d21c680ca1baf66e436772801bc) 的 `tgi/src/tgi/{mdp,world_model,training}` 实现。Go2 使用本机碰撞体和四个球足确定采样位置，保留其速度任务与真实 5 N 机身碰撞判据。回放、网络规格、损失与终止语义见[算法与任务说明](../README.md)。

PROOF及Z_RL原始BSD许可分别保留在 [LICENSE_proof](LICENSE_proof) 与 [LICENSE_z_rl](LICENSE_z_rl)。原世界模型／优化器的MIT许可继续保留如下。

以下保留所派生代码的MIT版权与许可原文：

```text
MIT License

Copyright (c) 2026 ETH Zurich (Terrain-Grounded Imagination)
Copyright (c) 2026 PartyMimic authors
Copyright (c) 2026 Fuyi Chen (BlinkMimic, from which this software is derived)
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which BlinkMimic is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
