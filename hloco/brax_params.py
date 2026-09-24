"""Save / load brax PPO policy parameters as plain numpy (no brax needed to read them)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _to_np(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _to_np(v) for k, v in tree.items()}
    return np.asarray(tree)


def export_policy(params: Any, obs_key: str = "state") -> dict[str, Any]:
    """Extract (normalizer mean/std for obs_key, actor MLP layers) from brax PPO params.

    brax PPO params are (normalizer_params, policy_params, value_params). The normalizer
    holds a dict of RunningStatisticsState per observation key.
    """
    normalizer, policy = params[0], params[1]
    mean = normalizer.mean[obs_key] if isinstance(normalizer.mean, dict) else normalizer.mean
    std = normalizer.std[obs_key] if isinstance(normalizer.std, dict) else normalizer.std
    policy = policy["params"] if "params" in policy else policy
    layer_names = sorted(policy.keys(), key=lambda s: int(s.split("_")[-1]))
    layers = [
        {"w": np.asarray(policy[n]["kernel"]), "b": np.asarray(policy[n]["bias"])}
        for n in layer_names
    ]
    return {"obs_key": obs_key, "mean": np.asarray(mean), "std": np.asarray(std), "layers": layers,
            "activation": "swish", "squash": "tanh"}


def save(path: Path, params: Any, extra: dict[str, Any] | None = None) -> None:
    import jax

    params = jax.device_get(params)
    blob = {"policy": export_policy(params), "brax_params": params, "extra": extra or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(blob, f)
    tmp.replace(path)


def load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def swish(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def np_policy(policy: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    """Deterministic action (numpy reference). brax: tanh(loc); RSL-RL: clip(mean, -1, 1)."""
    x = (obs.astype(np.float32) - policy["mean"]) / policy["std"]
    layers = policy["layers"]
    act = swish if policy.get("activation", "swish") == "swish" else elu
    for i, layer in enumerate(layers):
        x = x @ layer["w"] + layer["b"]
        if i < len(layers) - 1:
            x = act(x)
    if policy.get("squash", "tanh") == "tanh":
        return np.tanh(x[..., : layers[-1]["b"].shape[0] // 2])
    return np.clip(x, -1.0, 1.0)
