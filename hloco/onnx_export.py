"""Build a static ONNX graph (opset 17) for the deterministic actor: normalize -> MLP(swish) -> tanh."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17


def build_onnx(policy: dict[str, Any]) -> onnx.ModelProto:
    """policy: dict with mean, std, layers[{w, b}], activation, squash (see hloco.brax_params).

    activation: "swish" (brax default, x*sigmoid(x)) or "elu" (RSL-RL default).
    squash: "tanh" (brax NormalTanh head: the last layer outputs 2*act_dim = loc, scale;
    the deterministic action is tanh(loc)) or "clip" (RSL-RL: last layer outputs act_dim,
    the Playground torch wrapper clips actions to [-1, 1]).
    """
    layers = policy["layers"]
    activation = policy.get("activation", "swish")
    squash = policy.get("squash", "tanh")
    obs_dim = layers[0]["w"].shape[0]
    act_dim = layers[-1]["b"].shape[0] // 2 if squash == "tanh" else layers[-1]["b"].shape[0]
    inits = [
        numpy_helper.from_array(np.asarray(policy["mean"], np.float32).reshape(1, -1), "obs_mean"),
        numpy_helper.from_array(np.asarray(policy["std"], np.float32).reshape(1, -1), "obs_std"),
    ]
    nodes = [
        helper.make_node("Sub", ["obs", "obs_mean"], ["x_c"]),
        helper.make_node("Div", ["x_c", "obs_std"], ["h_in"]),
    ]
    cur = "h_in"
    for i, layer in enumerate(layers):
        w, b = f"w{i}", f"b{i}"
        inits += [numpy_helper.from_array(np.asarray(layer["w"], np.float32), w),
                  numpy_helper.from_array(np.asarray(layer["b"], np.float32), b)]
        nodes.append(helper.make_node("Gemm", [cur, w, b], [f"z{i}"]))
        cur = f"z{i}"
        if i < len(layers) - 1:
            if activation == "swish":
                nodes += [helper.make_node("Sigmoid", [cur], [f"s{i}"]),
                          helper.make_node("Mul", [cur, f"s{i}"], [f"a{i}"])]
                cur = f"a{i}"
            elif activation == "elu":
                nodes.append(helper.make_node("Elu", [cur], [f"a{i}"], alpha=1.0))
                cur = f"a{i}"
            else:
                raise ValueError(activation)
    if squash == "tanh":
        inits += [numpy_helper.from_array(np.array([0], np.int64), "sl_start"),
                  numpy_helper.from_array(np.array([act_dim], np.int64), "sl_end"),
                  numpy_helper.from_array(np.array([1], np.int64), "sl_axis")]
        nodes += [helper.make_node("Slice", [cur, "sl_start", "sl_end", "sl_axis"], ["loc"]),
                  helper.make_node("Tanh", ["loc"], ["action"])]
    elif squash == "clip":
        inits += [numpy_helper.from_array(np.array(-1.0, np.float32), "act_lo"),
                  numpy_helper.from_array(np.array(1.0, np.float32), "act_hi")]
        nodes.append(helper.make_node("Clip", [cur, "act_lo", "act_hi"], ["action"]))
    else:
        raise ValueError(squash)
    graph = helper.make_graph(
        nodes,
        "g1_actor",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, act_dim])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="humanoid-loco")
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def save_onnx(policy: dict[str, Any], path: Path) -> Path:
    model = build_onnx(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


class OnnxPolicy:
    """Single-observation CPU inference via onnxruntime."""

    def __init__(self, path: Path | str, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run(["action"], {"obs": obs.reshape(1, -1).astype(np.float32)})[0]
        return out[0]
