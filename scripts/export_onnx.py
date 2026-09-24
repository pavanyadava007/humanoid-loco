"""Export a trained actor (brax PPO or RSL-RL) to ONNX and check action parity against the framework policy.

Usage: python scripts/export_onnx.py --policy brax_dr
Writes checkpoints/<policy>/policy.onnx and results/onnx_parity_<policy>.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hloco import brax_params  # noqa: E402
from hloco.common import CKPT, ENV_NAME, RESULTS, write_json  # noqa: E402
from hloco.onnx_export import OPSET, OnnxPolicy, save_onnx  # noqa: E402


def jax_deterministic_actions(blob: dict, obs: np.ndarray) -> np.ndarray:
    """Actions from brax's own make_inference_fn (deterministic=True) for a batch of obs."""
    import functools

    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params

    cfg = locomotion_params.brax_ppo_config(ENV_NAME)
    net = functools.partial(ppo_networks.make_ppo_networks, **cfg.network_factory)
    obs_size = {"state": (obs.shape[1],), "privileged_state": (216,)}
    nets = net(obs_size, 29, preprocess_observations_fn=running_statistics.normalize)
    make_policy = ppo_networks.make_inference_fn(nets)
    params = blob["brax_params"]
    policy = make_policy((params[0], params[1]), deterministic=True)
    fake_priv = np.zeros((obs.shape[0], 216), np.float32)
    act, _ = jax.jit(policy)({"state": obs, "privileged_state": fake_priv}, jax.random.PRNGKey(0))
    return np.asarray(act)


def torch_rsl_actions(model_pt: Path, obs: np.ndarray) -> np.ndarray:
    """Actions straight from the RSL-RL checkpoint state_dict with torch ops (independent of the numpy export)."""
    import torch

    sd = torch.load(model_pt, map_location="cpu", weights_only=False)["model_state_dict"]
    x = torch.from_numpy(obs)
    x = (x - sd["actor_obs_normalizer._mean"]) / (sd["actor_obs_normalizer._std"] + 1e-2)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    for j, i in enumerate(idx):
        x = torch.nn.functional.linear(x, sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"])
        if j < len(idx) - 1:
            x = torch.nn.functional.elu(x)
    return torch.clamp(x, -1.0, 1.0).numpy()


def sample_obs(n: int, seed: int = 0) -> np.ndarray:
    """Realistic observations from CPU MuJoCo rollouts of the reset distribution (random actions)."""
    from hloco.mj_env import G1MjEnv, load_model

    env = G1MjEnv(load_model())
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        env.reset_random(rng)
        phase = np.array([0.0, np.pi])
        cmd = rng.uniform([-1, -0.5, -1], [1, 0.5, 1])
        last = np.zeros(29, np.float32)
        for _ in range(20):
            a = rng.uniform(-0.3, 0.3, 29).astype(np.float32)
            env.step_ctrl(env.motor_targets(a))
            out.append(env.obs(cmd, last, phase))
            last = a
            phase = np.fmod(phase + 0.16 + np.pi, 2 * np.pi) - np.pi
    return np.stack(out[:n]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    d = CKPT / args.policy
    obs = sample_obs(args.n)
    if (d / "params.pkl").exists():  # brax PPO
        blob = brax_params.load(d / "params.pkl")
        a_jax = jax_deterministic_actions(blob, obs)
        reference = "brax ppo_networks.make_inference_fn(deterministic=True), JAX"
    else:  # RSL-RL
        blob = brax_params.load(d / "policy.pkl")
        a_jax = torch_rsl_actions(d / "model.pt", obs)
        reference = "RSL-RL checkpoint state_dict evaluated with torch ops"
    pol = blob["policy"]
    onnx_path = save_onnx(pol, d / "policy.onnx")

    a_np = brax_params.np_policy(pol, obs)
    runner = OnnxPolicy(onnx_path)
    a_onnx = np.stack([runner(o) for o in obs])
    err_onnx_jax = np.abs(a_onnx - a_jax)
    err_np_jax = np.abs(a_np - a_jax)
    write_json(
        RESULTS / f"onnx_parity_{args.policy}.json",
        {
            "policy": args.policy,
            "checkpoint_step": blob["extra"].get("step"),
            "onnx_opset": OPSET,
            "onnx_file": str(onnx_path.name),
            "onnx_bytes": onnx_path.stat().st_size,
            "n_obs": int(args.n),
            "obs_source": "CPU MuJoCo rollouts from the reset distribution with random actions",
            "reference": reference,
            "max_abs_err_onnx_vs_reference": float(err_onnx_jax.max()),
            "mean_abs_err_onnx_vs_reference": float(err_onnx_jax.mean()),
            "max_abs_err_numpy_vs_reference": float(err_np_jax.max()),
        },
    )
    print(f"onnx vs reference max abs err {err_onnx_jax.max():.3e}")


if __name__ == "__main__":
    main()
