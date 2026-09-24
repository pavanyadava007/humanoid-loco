"""ONNX export must reproduce the framework policy's deterministic action."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools
import pickle

import numpy as np
import pytest

from hloco import brax_params
from hloco.common import CKPT, ENV_NAME
from hloco.onnx_export import OnnxPolicy, save_onnx

TOL = 1e-4


def _brax_random_params():
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground.config import locomotion_params

    cfg = locomotion_params.brax_ppo_config(ENV_NAME)
    obs_size = {"state": (103,), "privileged_state": (216,)}
    nets = functools.partial(ppo_networks.make_ppo_networks, **cfg.network_factory)(
        obs_size, 29, preprocess_observations_fn=running_statistics.normalize)
    key = jax.random.PRNGKey(0)
    pol = nets.policy_network.init(key)
    norm = running_statistics.init_state({k: jax.ShapeDtypeStruct(v, np.float32) for k, v in obs_size.items()})
    rng = np.random.default_rng(0)
    # non-trivial normalizer statistics
    norm = norm.replace(
        mean={k: rng.normal(size=v).astype(np.float32) for k, v in obs_size.items()},
        std={k: rng.uniform(0.5, 2.0, size=v).astype(np.float32) for k, v in obs_size.items()},
    )
    make_policy = ppo_networks.make_inference_fn(nets)
    return (norm, pol, None), make_policy


def test_brax_onnx_matches_jax(tmp_path):
    import jax

    params, make_policy = _brax_random_params()
    exported = brax_params.export_policy(jax.device_get(params))
    path = save_onnx(exported, tmp_path / "p.onnx")
    obs = np.random.default_rng(1).normal(size=(64, 103)).astype(np.float32)
    ref, _ = make_policy((params[0], params[1]), deterministic=True)(
        {"state": obs, "privileged_state": np.zeros((64, 216), np.float32)}, jax.random.PRNGKey(0))
    ref = np.asarray(ref)
    onnx_pol = OnnxPolicy(path)
    got = np.stack([onnx_pol(o) for o in obs])
    assert np.abs(got - ref).max() < TOL
    assert np.abs(brax_params.np_policy(exported, obs) - ref).max() < TOL


def test_elu_clip_export_matches_numpy(tmp_path):
    rng = np.random.default_rng(2)
    dims = [103, 64, 32, 29]
    pol = {"mean": rng.normal(size=103).astype(np.float32), "std": rng.uniform(0.5, 2, 103).astype(np.float32),
           "layers": [{"w": rng.normal(size=(a, b)).astype(np.float32) * 0.3, "b": rng.normal(size=b).astype(np.float32)}
                      for a, b in zip(dims[:-1], dims[1:], strict=True)],
           "activation": "elu", "squash": "clip"}
    path = save_onnx(pol, tmp_path / "r.onnx")
    obs = rng.normal(size=(32, 103)).astype(np.float32)
    got = np.stack([OnnxPolicy(path)(o) for o in obs])
    assert np.abs(got - brax_params.np_policy(pol, obs)).max() < TOL


@pytest.mark.parametrize("name", ["brax_dr", "brax_nodr", "rsl_dr"])
def test_trained_onnx_matches_numpy(name):
    d = CKPT / name
    blob_path = d / ("params.pkl" if (d / "params.pkl").exists() else "policy.pkl")
    if not (d / "policy.onnx").exists() or not blob_path.exists():
        pytest.skip(f"{name} not trained/exported")
    with open(blob_path, "rb") as fh:
        pol = pickle.load(fh)["policy"]
    obs = np.random.default_rng(3).normal(size=(64, 103)).astype(np.float32) * 0.5
    runner = OnnxPolicy(d / "policy.onnx")
    got = np.stack([runner(o) for o in obs])
    assert np.abs(got - brax_params.np_policy(pol, obs)).max() < TOL
