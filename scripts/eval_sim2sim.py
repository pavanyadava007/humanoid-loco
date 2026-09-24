"""Sim-to-sim evaluation: ONNX policy closed-loop in plain CPU MuJoCo (not MJX).

Usage:
  python scripts/eval_sim2sim.py --policy brax_dr --episodes 100
Reads checkpoints/<policy>/policy.onnx, writes results/sim2sim_<policy>.json.
Episode i uses seed BASE_SEED + i in every condition and for every policy (paired design).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from hloco.common import CKPT, RESULTS, write_json  # noqa: E402
from hloco.mj_env import G1MjEnv, Perturbation  # noqa: E402
from hloco.stats import mean_ci, wilson  # noqa: E402

BASE_SEED = 10_000
EPISODE_STEPS = 1000  # 20 s at the 50 Hz policy rate (training episode length)
WARMUP_STEPS = 50  # tracking error is averaged from t = 1 s on

CONDITIONS = [
    Perturbation("nominal"),
    Perturbation("friction_x0.5", friction_scale=0.5),
    Perturbation("friction_x1.5", friction_scale=1.5),
    Perturbation("torso_mass_+3kg", torso_mass_delta_kg=3.0),
    Perturbation("torso_mass_-3kg", torso_mass_delta_kg=-3.0),
    Perturbation("pushes", push=True),
    Perturbation("action_delay_1step", action_delay_steps=1),
    Perturbation("action_delay_2steps", action_delay_steps=2),
    Perturbation("obs_noise_train_level", obs_noise=1.0),
    Perturbation("accurate_solver", solver="accurate"),
]
COND_BY_NAME = {c.name: c for c in CONDITIONS}

_W: dict = {}


def _init_worker(mjb_path: str, onnx_path: str) -> None:
    import mujoco

    from hloco.onnx_export import OnnxPolicy

    _W["model"] = mujoco.MjModel.from_binary_path(mjb_path)
    _W["policy"] = OnnxPolicy(onnx_path, threads=1)


def sample_command(rng: np.random.Generator) -> np.ndarray:
    """Command ranges of the training config (lin_vel_x, lin_vel_y, ang_vel_yaw), no zeroing."""
    return np.array([rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5), rng.uniform(-1.0, 1.0)])


def run_episode(model, policy, pert: Perturbation, seed: int, steps: int = EPISODE_STEPS) -> dict:
    env = G1MjEnv(copy.deepcopy(model), pert)  # fresh copy: perturbations never leak
    rng = np.random.default_rng(seed)
    env.reset_random(rng)
    cmd = sample_command(rng)
    gait_freq = rng.uniform(1.25, 1.5)
    phase_dt = 2 * np.pi * env.ctrl_dt * gait_freq
    phase = np.array([0.0, np.pi])
    push_rng = np.random.default_rng(seed + 7_000_000)
    noise_rng = np.random.default_rng(seed + 9_000_000)
    next_push = int(round(push_rng.uniform(*pert.push_interval_s) / env.ctrl_dt))

    info_last_act = np.zeros(env.nu, np.float32)
    delay = deque([np.zeros(env.nu, np.float32)] * pert.action_delay_steps)
    obs = env.obs(cmd, info_last_act, phase, noise_rng)
    lin_err, yaw_err = [], []
    fell, t_end, n_push = False, steps, 0
    for t in range(steps):
        act = policy(obs)
        if pert.push and t == next_push:
            theta = push_rng.uniform(0, 2 * np.pi)
            mag = push_rng.uniform(*pert.push_magnitude)
            env.data.qvel[0:2] += mag * np.array([np.cos(theta), np.sin(theta)])
            next_push = t + int(round(push_rng.uniform(*pert.push_interval_s) / env.ctrl_dt))
            n_push += 1
        delay.append(act)
        applied = delay.popleft()
        env.step_ctrl(env.motor_targets(applied))
        obs = env.obs(cmd, info_last_act, phase, noise_rng)
        info_last_act = act
        phase = np.fmod(phase + phase_dt + np.pi, 2 * np.pi) - np.pi
        if env.terminated():
            fell, t_end = True, t + 1
            break
        if t >= WARMUP_STEPS:
            v = env.sensor("local_linvel_pelvis")
            w = env.sensor("gyro_pelvis")
            lin_err.append(float(np.linalg.norm(cmd[:2] - v[:2])))
            yaw_err.append(float(abs(cmd[2] - w[2])))
    return {
        "seed": seed,
        "command": cmd.tolist(),
        "fell": fell,
        "survival_s": t_end * env.ctrl_dt,
        "lin_vel_err": float(np.mean(lin_err)) if lin_err else None,
        "yaw_rate_err": float(np.mean(yaw_err)) if yaw_err else None,
        "pushes": n_push,
    }


def _job(args: tuple[str, int]) -> tuple[str, dict]:
    cond, seed = args
    return cond, run_episode(_W["model"], _W["policy"], COND_BY_NAME[cond], seed)


def summarize(eps: list[dict]) -> dict:
    n = len(eps)
    k = sum(e["fell"] for e in eps)
    lo, hi = wilson(k, n)
    surv = [e["survival_s"] for e in eps]
    lin = [e["lin_vel_err"] for e in eps if e["lin_vel_err"] is not None]
    yaw = [e["yaw_rate_err"] for e in eps if e["yaw_rate_err"] is not None]
    m_lin, lin_lo, lin_hi = mean_ci(lin)
    m_yaw, yaw_lo, yaw_hi = mean_ci(yaw)
    m_s, s_lo, s_hi = mean_ci(surv)
    return {
        "episodes": n,
        "falls": k,
        "fall_rate": k / n if n else None,
        "fall_rate_wilson95": [lo, hi],
        "survival_s_mean": m_s,
        "survival_s_ci95": [s_lo, s_hi],
        "lin_vel_err_mean": m_lin,
        "lin_vel_err_ci95": [lin_lo, lin_hi],
        "yaw_rate_err_mean": m_yaw,
        "yaw_rate_err_ci95": [yaw_lo, yaw_hi],
        "tracking_episodes": len(lin),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="checkpoint dir name, e.g. brax_dr")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    ap.add_argument("--conditions", nargs="*", default=[c.name for c in CONDITIONS])
    args = ap.parse_args()

    import mujoco

    from hloco.mj_env import load_model

    onnx_path = CKPT / args.policy / "policy.onnx"
    if not onnx_path.exists():
        raise SystemExit(f"missing {onnx_path}; run scripts/export_onnx.py first")
    model = load_model()
    tmp = Path(tempfile.mkdtemp()) / "g1_flat.mjb"
    mujoco.mj_saveModel(model, str(tmp), None)

    jobs = [(c, BASE_SEED + i) for c in args.conditions for i in range(args.episodes)]
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(tmp), str(onnx_path))) as pool:
        out = pool.map(_job, jobs, chunksize=4)
    wall = time.time() - t0
    per_cond: dict[str, list[dict]] = {c: [] for c in args.conditions}
    for c, ep in out:
        per_cond[c].append(ep)
    for c in per_cond:
        per_cond[c].sort(key=lambda e: e["seed"])

    write_json(
        RESULTS / f"sim2sim_{args.policy}.json",
        {
            "policy": args.policy,
            "onnx": str(onnx_path.name),
            "simulator": f"MuJoCo {mujoco.__version__} CPU (mujoco python bindings, not MJX)",
            "control_rate_hz": 50,
            "episode_steps": EPISODE_STEPS,
            "warmup_steps_excluded_from_tracking": WARMUP_STEPS,
            "base_seed": BASE_SEED,
            "fall_definition": "Playground G1 termination: torso up-vector z < 0, foot-foot or foot-shin contact, or NaN state",
            "command_sampling": "per episode uniform lin_vel_x [-1,1], lin_vel_y [-0.5,0.5], yaw [-1,1] m/s, rad/s; held constant",
            "initial_state": "Joystick.reset distribution (keyframe knees_bent, xy +-0.5 m, yaw, joint x U(0.5,1.5), base vel U(-0.5,0.5))",
            "wall_clock_s": wall,
            "workers": args.workers,
            "conditions": {c: asdict(COND_BY_NAME[c]) for c in args.conditions},
            "summary": {c: summarize(per_cond[c]) for c in args.conditions},
            "episodes": per_cond,
        },
    )
    print(f"wrote results/sim2sim_{args.policy}.json in {wall:.1f}s")


if __name__ == "__main__":
    main()
