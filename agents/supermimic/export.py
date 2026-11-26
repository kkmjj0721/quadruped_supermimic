# Copyright (c) 2026-2027 zh
"""内容：导出含本体／高程双编码器、回合mask及动作网络的三输入教师策略。"""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import torch

from .policy import Mlp


def _prepare_policy(actor, iteration, metadata):
    """冻结部署副本并生成导出元数据。"""
    source_precision = metadata["source_network_precision"]
    module = deepcopy(actor).cpu().eval().requires_grad_(False)
    # 部署副本统一使用 FP32，训练策略的精度开关保持原值。
    for layer in module.modules():
        if isinstance(layer, Mlp):
            layer.use_bf16 = False
    encoder = {
        "type": "dual_mlp",
        "proprioception": {
            "input": "normalized_history", "hidden_dims": list(module.prop_encoder_hidden),
            "latent_dim": module.prop_latent_dim, "layer_norm": "pre_activation", "activation": "elu",
            "velocity": {"columns": [0, 1, 2], "frame": "body", "components": ["vx", "vy", "vz"],
                         "units": "m/s", "time": "current", "normalization": "none"},
        },
        "heightmap": {
            "input": "heightmap", "input_dim": module.heightmap_dim,
            "hidden_dims": list(module.height_encoder_hidden), "latent_dim": module.height_latent_dim,
            "layer_norm": "pre_activation", "activation": "elu", "output_activation": "tanh",
            "mask_input": "exteroceptive_mask", "mask_stage": "after_tanh",
        },
    }
    information = {
        **metadata, "format": "supermimic_proof_teacher_policy_v1",
        "iteration": iteration,
        "input_names": ["obs_history", "heightmap", "exteroceptive_mask"],
        "output_name": "actions", "dtype": "float32",
        "source_network_precision": source_precision, "export_network_precision": "fp32",
        "input_shapes": {"obs_history": [None, module.history_length * module.observation_dim],
                         "heightmap": [None, module.heightmap_dim], "exteroceptive_mask": [None, 1]},
        "output_shape": [None, module.action_dim], "observation_dim": module.observation_dim,
        "heightmap_dim": module.heightmap_dim,
        "history_length": module.history_length, "history_order": module.history_order,
        "encoder_history_order": module.encoder_history_order,
        "history_reset": "repeat_initial_frame",
        "normalization": {"statistics_in_graph": True, "proprioception": "empirical_population_std",
                          "formula": "(x-mean)/(sqrt(var)+1e-2)", "initial_count": 0,
                          "current_frame": "select_normalized_term_history", "heightmap": "none", "mask": "none"},
        "action_semantics": "unscaled_joint_position_offsets", "action_transform": "identity",
        "actor_input_order": module.input_order,
        "actor_input_dim": module.actor_input_dim,
        "encoder": encoder,
    }
    return module, information


def _write_onnx(module, path, information):
    """把完整策略与元数据写入ONNX文件。"""
    import onnx

    inputs = (torch.zeros(1, module.history_length * module.observation_dim),
              torch.zeros(1, module.heightmap_dim), torch.ones(1, 1))
    torch.onnx.export(module, inputs,
                      str(path), export_params=True, opset_version=18, dynamo=False,
                      input_names=["obs_history", "heightmap", "exteroceptive_mask"],
                      output_names=["actions"], external_data=False,
                      dynamic_axes={"obs_history": {0: "batch"}, "heightmap": {0: "batch"},
                                    "exteroceptive_mask": {0: "batch"}, "actions": {0: "batch"}})
    model = onnx.load(path)
    for name, value in information.items():
        field = model.metadata_props.add()
        field.key, field.value = name, json.dumps(value, ensure_ascii=False, allow_nan=False)
    onnx.save(model, path)


def export_policy(actor, directory, *, iteration: int, metadata):
    """直接导出PT／ONNX／JSON轮次文件，并更新固定名称副本。"""
    module, information = _prepare_policy(actor, iteration, metadata)

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {"pt": destination / f"policy_{iteration}.pt", "onnx": destination / f"policy_{iteration}.onnx",
             "metadata": destination / f"policy_{iteration}.json"}
    torch.jit.script(module).save(str(paths["pt"]))
    _write_onnx(module, paths["onnx"], information)
    paths["metadata"].write_text(json.dumps(information, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    for kind, path in paths.items():
        filename = "policy.json" if kind == "metadata" else f"policy.{kind}"
        shutil.copyfile(path, destination / filename)
    return paths

