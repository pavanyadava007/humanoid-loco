"""The sim-to-sim evaluation must be exactly reproducible for a fixed seed."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np

from hloco.common import ROOT
from hloco.mj_env import Perturbation, load_model
from hloco.onnx_export import OnnxPolicy, save_onnx
from hloco.stats import wilson


def _eval_module():
    spec = importlib.util.spec_from_file_location("eval_sim2sim", ROOT / "scripts" / "eval_sim2sim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_sim2sim"] = mod
    spec.loader.exec_module(mod)
    return mod


def _small_policy(tmp_path):
    rng = np.random.default_rng(0)
    dims = [103, 32, 58]
    pol = {"mean": np.zeros(103, np.float32), "std": np.ones(103, np.float32) * 5,
           "layers": [{"w": rng.normal(size=(a, b)).astype(np.float32) * 0.05, "b": np.zeros(b, np.float32)}
                      for a, b in zip(dims[:-1], dims[1:], strict=True)],
           "activation": "swish", "squash": "tanh"}
    return OnnxPolicy(save_onnx(pol, tmp_path / "small.onnx"))


def test_same_seed_same_episode(tmp_path):
    ev = _eval_module()
    model = load_model()
    pol = _small_policy(tmp_path)
    for pert in (Perturbation("pushes", push=True), Perturbation("action_delay_2steps", action_delay_steps=2),
                 Perturbation("obs_noise", obs_noise=1.0)):
        a = ev.run_episode(model, pol, pert, seed=123, steps=150)
        b = ev.run_episode(model, pol, pert, seed=123, steps=150)
        assert a == b
    c = ev.run_episode(model, pol, Perturbation(), seed=124, steps=150)
    assert c["command"] != a["command"]


def test_perturbation_does_not_leak(tmp_path):
    ev = _eval_module()
    model = load_model()
    mass0, fr0 = model.body_mass.copy(), model.pair_friction.copy()
    pol = _small_policy(tmp_path)
    ev.run_episode(model, pol, Perturbation("x", friction_scale=0.5, torso_mass_delta_kg=3.0), seed=1, steps=5)
    assert np.array_equal(model.body_mass, mass0) and np.array_equal(model.pair_friction, fr0)


def test_wilson_known_values():
    lo, hi = wilson(0, 100)
    assert abs(lo) < 1e-12 and abs(hi - 0.0370) < 1e-3
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
