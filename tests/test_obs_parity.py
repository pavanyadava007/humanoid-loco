"""The CPU MuJoCo observation must equal the MJX (Playground) observation for the same state."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def mjx_states():
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco_playground import registry

    from hloco.common import ENV_NAME

    cfg = registry.get_default_config(ENV_NAME)
    cfg.noise_config.level = 0.0
    env = registry.load(ENV_NAME, config=cfg)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    fwd = jax.jit(lambda d: mjx.forward(env.mjx_model, d))
    get_obs = jax.jit(lambda d, info, c: env._get_obs(d, dict(info), c)["state"])
    rng = np.random.default_rng(0)
    out = []
    for k in range(3):
        s = reset(jax.random.PRNGKey(k))
        for _ in range(5 + 7 * k):
            a = jp.asarray(rng.uniform(-0.3, 0.3, env.action_size), jp.float32)
            s = step(s, a)
        d = fwd(s.data)  # make sensors/kinematics consistent with qpos/qvel
        contact = jp.zeros(2, bool)
        obs = get_obs(d, s.info, contact)
        out.append({
            "qpos": np.asarray(d.qpos),
            "qvel": np.asarray(d.qvel),
            "command": np.asarray(s.info["command"]),
            "last_act": np.asarray(s.info["last_act"]),
            "phase": np.asarray(s.info["phase"]),
            "obs": np.asarray(obs),
        })
    return out


def test_obs_matches_mjx(mjx_states):
    from hloco.mj_env import G1MjEnv, load_model

    env = G1MjEnv(load_model())
    for st in mjx_states:
        env.set_state(st["qpos"], st["qvel"])
        o = env.obs(st["command"], st["last_act"], st["phase"])
        assert o.shape == st["obs"].shape == (103,)
        err = np.abs(o - st["obs"])
        # float32 (MJX) vs float64 (MuJoCo) arithmetic on identical state
        assert err.max() < 1e-4, (err.max(), int(err.argmax()))


def test_termination_flags_nominal():
    from hloco.mj_env import G1MjEnv, load_model

    env = G1MjEnv(load_model())
    m = env.model
    env.set_state(m.keyframe("knees_bent").qpos, np.zeros(m.nv))
    assert not env.terminated()
