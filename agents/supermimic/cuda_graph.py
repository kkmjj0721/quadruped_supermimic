# Copyright (c) 2026-2027 zh
"""内容：用静态输入和可捕获优化器执行世界模型、策略与价值更新，保持原训练预算。"""

from __future__ import annotations

from contextlib import contextmanager
import gc

import torch

from .learner import Learner
from .optim import MuonAdam


# ---------- 静态输入与实际更新预热 ----------

@contextmanager
def _capture_without_gc():
    """仅捕获期间暂停循环回收，避免析构Triton内核；始终恢复原开关。"""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class CapturedUpdate:
    """前三次真实更新用于侧流预热，第四次捕获并重放；从不额外训练。"""

    def __init__(self, function, device: torch.device, *, warmup: int = 3):
        self.function, self.device, self.warmup = function, torch.device(device), warmup
        self.calls = self.replays = 0
        self.graph = self.outputs = None
        self.stream = torch.cuda.Stream(device=self.device)

    def invalidate(self):
        """释放旧捕获而保留专用流，避免流池轮转后影响其它存活图的工作区。"""
        self.calls = self.replays = 0
        self.graph = self.outputs = None

    def __call__(self):
        # torch.compile、无显式device的算子和图创建均使用同一目标设备。
        with torch.cuda.device(self.device):
            self.calls += 1
            current = torch.cuda.current_stream(self.device)
            if self.calls <= self.warmup:
                self.stream.wait_stream(current)
                with torch.cuda.stream(self.stream):
                    outputs = self.function()
                current.wait_stream(self.stream)
                return outputs
            if self.graph is None:
                torch.cuda.synchronize(self.device)
                with _capture_without_gc():
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.outputs = self.function()
                # 捕获只记录 CUDA 工作；下面这次 replay 才是本次实际更新。
                print(f"[INFO] Captured CUDA update graph: {self.function.__name__}", flush=True)
            self.graph.replay()
            self.replays += 1
            return self.outputs


# ---------- 可捕获更新 ----------

class CudaGraphLearner(Learner):
    """复用唯一学习实现，整轮采样与更新由runner捕获。"""

    def __init__(self, task, actor, model, critic, config, execution=None):
        self.device = next(model.parameters()).device
        self.graph_generation = 0
        self.model_graph_generation = 0
        super().__init__(task, actor, model, critic, config, execution)
        self.model_optimizer = self._make_optimizer("model", capturable=True)

    def reset_policy_optimizers(self):
        """护栏替换优化器后让runner重新捕获策略与价值更新。"""
        self.actor_optimizer = self._make_optimizer("actor", capturable=True)
        self.critic_optimizer = self._make_optimizer("critic", capturable=True)
        self.graph_generation += 1

    def schedule(self, iteration: int):
        rates = [(group, group["lr"]) for optimizer in
                 (self.actor_optimizer, self.model_optimizer, self.critic_optimizer) for group in optimizer.param_groups]
        super().schedule(iteration)
        for group, reference in rates:
            reference.fill_(group["lr"])
            group["lr"] = reference

    def load_optimizer_state_dict(self, name, data):
        """恢复时保留设备学习率地址，避免重放使用旧常量或旧优化器状态。"""
        optimizer = getattr(self, name + "_optimizer")
        if isinstance(optimizer, MuonAdam):
            optimizer.load_state_dict(data)
        else:
            references = [group["lr"] for group in optimizer.param_groups]
            state = {**data, "param_groups": [{**group, "fused": True, "capturable": True}
                                              for group in data["param_groups"]]}
            optimizer.load_state_dict(state)
            for group, reference in zip(optimizer.param_groups, references, strict=True):
                reference.copy_(torch.as_tensor(group["lr"], device=self.device))
                group["lr"] = reference
        # 加载替换状态存储，按受影响优化器失效图地址。
        if name == "model":
            self.model_graph_generation += 1
        else:
            self.graph_generation += 1
