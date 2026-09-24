"""Load an exported actor (brax or RSL-RL) and rebuild it in JAX."""

from __future__ import annotations

import pickle

from hloco.common import CKPT


def load_actor(name: str) -> dict:
    for fn in ("params.pkl", "policy.pkl"):
        p = CKPT / name / fn
        if p.exists():
            with open(p, "rb") as f:
                blob = pickle.load(f)
            return {"policy": blob["policy"], "step": blob["extra"].get("step")}
    raise FileNotFoundError(f"no params.pkl or policy.pkl in {CKPT / name}")


def jax_actor(policy: dict):
    """Deterministic actor as a JAX function of the 103-dim 'state' observation."""
    import jax
    import jax.numpy as jp

    mean, std = jp.asarray(policy["mean"]), jp.asarray(policy["std"])
    layers = [(jp.asarray(ly["w"]), jp.asarray(ly["b"])) for ly in policy["layers"]]
    act = jax.nn.swish if policy["activation"] == "swish" else jax.nn.elu

    def f(obs):
        x = (obs - mean) / std
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if i < len(layers) - 1:
                x = act(x)
        if policy["squash"] == "tanh":
            return jp.tanh(x[..., : layers[-1][1].shape[0] // 2])
        return jp.clip(x, -1.0, 1.0)

    return f
