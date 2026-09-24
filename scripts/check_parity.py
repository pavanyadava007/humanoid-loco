"""MJX vs CPU MuJoCo parity diagnostics.

1. Observation construction: same state -> same 103-dim policy observation.
2. One-control-step dynamics gap: same state + same motor targets, 10 substeps in MJX (JAX impl)
   and in CPU MuJoCo -> difference in qpos/qvel. This quantifies the simulator gap the policy
   meets in the sim-to-sim evaluation (it is not expected to be zero).
Writes results/obs_parity.json.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hloco.common import ENV_NAME, RESULTS, write_json  # noqa: E402


def main(n_states: int = 20) -> None:
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env

    from hloco.mj_env import G1MjEnv, load_model

    cfg = registry.get_default_config(ENV_NAME)
    cfg.noise_config.level = 0.0
    env = registry.load(ENV_NAME, config=cfg)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    fwd = jax.jit(lambda d: mjx.forward(env.mjx_model, d))
    get_obs = jax.jit(lambda d, info: env._get_obs(d, dict(info), jp.zeros(2, bool))["state"])
    ctrl_step = jax.jit(lambda d, u: mjx_env.step(env.mjx_model, d, u, env.n_substeps))

    cpu = G1MjEnv(load_model())
    rng = np.random.default_rng(0)
    obs_err, dq, dv = [], [], []
    for k in range(n_states):
        s = reset(jax.random.PRNGKey(100 + k))
        for _ in range(int(rng.integers(3, 30))):
            s = step(s, jp.asarray(rng.uniform(-0.3, 0.3, 29), jp.float32))
        d = fwd(s.data)
        o_mjx = np.asarray(get_obs(d, s.info))
        cpu.set_state(np.asarray(d.qpos), np.asarray(d.qvel))
        o_cpu = cpu.obs(np.asarray(s.info["command"]), np.asarray(s.info["last_act"]), np.asarray(s.info["phase"]))
        obs_err.append(float(np.abs(o_cpu - o_mjx).max()))

        u = np.asarray(env._default_pose) + rng.uniform(-0.3, 0.3, 29) * 0.5
        d2 = ctrl_step(d, jp.asarray(u, jp.float32))
        cpu.set_state(np.asarray(d.qpos), np.asarray(d.qvel))
        cpu.step_ctrl(u)
        dq.append(float(np.abs(np.asarray(d2.qpos) - cpu.data.qpos).max()))
        dv.append(float(np.abs(np.asarray(d2.qvel) - cpu.data.qvel).max()))

    write_json(
        RESULTS / "obs_parity.json",
        {
            "n_states": n_states,
            "obs_max_abs_err": max(obs_err),
            "obs_max_abs_err_per_state": obs_err,
            "one_ctrl_step_qpos_max_abs_diff": {"max": max(dq), "median": float(np.median(dq))},
            "one_ctrl_step_qvel_max_abs_diff": {"max": max(dv), "median": float(np.median(dv))},
            "note": "MJX jax impl in float32 (JAX_PLATFORMS=cpu) vs MuJoCo C float64; states from random-action rollouts",
        },
    )
    print("obs max err", max(obs_err), "qpos diff median", np.median(dq), "qvel diff median", np.median(dv))


if __name__ == "__main__":
    main()
