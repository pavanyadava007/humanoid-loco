"""Render short MP4s of a trained policy.

Two videos per policy:
  media/<policy>_mjx.mp4        rollout in the MJX training env (JAX policy, training obs noise on, pushes disabled)
  media/<policy>_mujoco_cpu.mp4 rollout in plain CPU MuJoCo with the ONNX policy (sim-to-sim)
Both follow the same scripted command schedule. Writes results/render_<policy>.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hloco.common import CKPT, ENV_NAME, MEDIA, RESULTS, write_json  # noqa: E402

# (start_s, vx, vy, yaw_rate)
SCHEDULE = [(0.0, 0.0, 0.0, 0.0), (1.0, 0.8, 0.0, 0.0), (4.0, 0.5, 0.0, 0.8), (7.0, 0.0, 0.4, 0.0),
            (9.0, -0.5, 0.0, 0.0), (11.0, 0.0, 0.0, 0.0)]
DURATION_S = 12.0
CTRL_DT = 0.02


def command_at(t: float) -> np.ndarray:
    cmd = SCHEDULE[0][1:]
    for s in SCHEDULE:
        if t >= s[0]:
            cmd = s[1:]
    return np.array(cmd, np.float32)


def render_qpos(model: mujoco.MjModel, qpos_seq: list[np.ndarray], path: Path, fps: int = 25) -> int:
    import mediapy

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("pelvis").id
    cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 135.0
    frames = []
    for q in qpos_seq:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=fps)
    return len(frames)


def rollout_mjx(policy_name: str, steps: int) -> tuple[list[np.ndarray], bool]:
    import jax
    import jax.numpy as jp
    from mujoco_playground import registry

    from hloco.policy import jax_actor, load_actor

    env = registry.load(ENV_NAME, config_overrides={"push_config.enable": False})
    actor = jax.jit(jax_actor(load_actor(policy_name)["policy"]))
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    qs, fell = [np.asarray(state.data.qpos)], False
    for i in range(steps):
        state.info["command"] = jp.asarray(command_at(i * CTRL_DT))
        state.info["step"] = jp.array(0, jp.int32)  # keep the scripted command (env resamples after 500 steps)
        state = step(state, actor(state.obs["state"]))
        qs.append(np.asarray(state.data.qpos))
        if float(state.done) > 0:
            fell = True
            break
    return qs, fell


def rollout_cpu(policy_name: str, model: mujoco.MjModel, steps: int) -> tuple[list[np.ndarray], bool]:
    from hloco.mj_env import G1MjEnv
    from hloco.onnx_export import OnnxPolicy

    env = G1MjEnv(model)
    pol = OnnxPolicy(CKPT / policy_name / "policy.onnx")
    env.set_state(env.init_q, np.zeros(model.nv), ctrl=env.default_pose)
    phase = np.array([0.0, np.pi])
    phase_dt = 2 * np.pi * CTRL_DT * 1.375
    last = np.zeros(env.nu, np.float32)
    obs = env.obs(command_at(0.0), last, phase)
    qs, fell = [env.data.qpos.copy()], False
    for i in range(steps):
        act = pol(obs)
        env.step_ctrl(env.motor_targets(act))
        obs = env.obs(command_at((i + 1) * CTRL_DT), last, phase)
        last = act
        phase = np.fmod(phase + phase_dt + np.pi, 2 * np.pi) - np.pi
        qs.append(env.data.qpos.copy())
        if env.terminated():
            fell = True
            break
    return qs, fell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--skip-mjx", action="store_true")
    args = ap.parse_args()
    from hloco.mj_env import load_model

    steps = int(DURATION_S / CTRL_DT)
    model = load_model()
    out = {"policy": args.policy, "schedule_s_vx_vy_yaw": SCHEDULE, "duration_s": DURATION_S, "videos": {}}
    if not args.skip_mjx:
        qs, fell = rollout_mjx(args.policy, steps)
        n = render_qpos(model, qs[::2], MEDIA / f"{args.policy}_mjx.mp4")
        out["videos"]["mjx"] = {"file": f"media/{args.policy}_mjx.mp4", "frames": n, "fell": fell,
                                "note": "MJX env, deterministic actor in JAX, training obs noise on, pushes disabled"}
    if (CKPT / args.policy / "policy.onnx").exists():
        qs, fell = rollout_cpu(args.policy, model, steps)
        n = render_qpos(model, qs[::2], MEDIA / f"{args.policy}_mujoco_cpu.mp4")
        out["videos"]["mujoco_cpu"] = {"file": f"media/{args.policy}_mujoco_cpu.mp4", "frames": n, "fell": fell,
                                       "note": "plain CPU MuJoCo, ONNX policy, clean obs, no pushes"}
    write_json(RESULTS / f"render_{args.policy}.json", out)
    print(out["videos"])


if __name__ == "__main__":
    main()
