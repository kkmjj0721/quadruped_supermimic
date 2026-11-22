# 算法目录

本项目维护的算法统一放在 `agents/`。当前仅保留 `agents/supermimic/`；任务观测、奖励和终止规则留在 `src/tasks/`，机器人与训练配置组合留在 `src/config/`。

```text
agents/
├── supermimic/       # 世界模型、连续回放与BPTT学习
│   ├── config.py         # SuperMimicConfig、ExecutionConfig
│   ├── interfaces.py     # TaskState、EnvStep、AlgorithmSpec
│   ├── world_model.py    # 归一化与世界模型
│   ├── policy.py         # 混合专家 actor、critic
│   ├── buffer.py         # 种子、窗口、回放与归档
│   ├── imagine.py        # 想象轨迹与回报
│   ├── learner.py        # 梯度裁剪与更新
│   ├── runner.py         # 训练循环、保存与续训
│   ├── export.py         # TorchScript／ONNX 导出
│   ├── logging.py        # TensorBoard 与终端日志
│   └── NOTICE.md
└── __init__.py
```

## SuperMimic 混合专家历史策略

默认 Go2 任务使用 `agents.supermimic`：go2_rl_gym 的八专家历史编码器加 ELU 动作网络、SiLU 世界模型和短时域 BPTT。无教师或蒸馏，部署只输入五帧本体历史；编码器输出32维潜特征和3维当前机身线速度估计，仅潜特征做L2归一化，速度由真实回放同一时刻真值监督。配置与训练、播放、验证入口统一见 [Go2任务说明](../docs/go2_supermimic.md)，来源与迁移差异见 [supermimic/NOTICE.md](supermimic/NOTICE.md)。
