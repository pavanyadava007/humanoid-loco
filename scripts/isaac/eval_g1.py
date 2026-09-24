"""Evaluate an exported Isaac Lab G1 policy inside Isaac Sim (PhysX) under perturbations.

Within Isaac Sim (PhysX), not sim-to-sim: the policy is evaluated in the simulator it was trained in.
Runs in .venv-isaac. Each condition runs in its own process (startup events such as friction and mass
are applied when the env is created), with one 20 s episode per parallel env:

    OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/isaac/eval_g1.py --all --name g1_flat --episodes 500

--all spawns one subprocess per condition and merges them into results/isaac_eval_<name>.json.

Protocol (mirrors the CPU MuJoCo table in spirit, not in detail):
- task env config of training (Isaac-Velocity-Flat-G1-v0), with observation noise off and the task's
  push/external-force events off unless the condition turns them on;
- one constant command per episode, sampled uniformly from the task's own training ranges (flat G1: vx 0 to 1,
  vy -0.5 to 0.5, wz -1 to 1; rough G1: vy fixed at 0); heading control off so the yaw-rate command stays constant;
- fall = the task's own non-timeout termination (torso_link contact force > 1 N) before 20 s;
- tracking error = mean over surviving steps after 1 s of |cmd_xy - base lin vel xy (base frame)| and
  |cmd_wz - base yaw rate|.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hloco.stats import mean_ci, wilson  # noqa: E402

EPISODE_STEPS = 1000  # 20 s at 50 Hz
WARMUP_STEPS = 50
BASE_SEED = 10_000


@dataclass(frozen=True)
class Condition:
    name: str
    friction_scale: float = 1.0
    torso_mass_delta_kg: float = 0.0
    push: bool = False
    push_interval_s: tuple[float, float] = (2.0, 4.0)
    push_magnitude: tuple[float, float] = (0.5, 1.5)
    action_delay_steps: int = 0
    obs_noise: bool = False


CONDITIONS = [
    Condition("nominal"),
    Condition("friction_x0.5", friction_scale=0.5),
    Condition("friction_x1.5", friction_scale=1.5),
    Condition("torso_mass_+3kg", torso_mass_delta_kg=3.0),
    Condition("torso_mass_-3kg", torso_mass_delta_kg=-3.0),
    Condition("pushes", push=True),
    Condition("action_delay_1step", action_delay_steps=1),
    Condition("action_delay_2steps", action_delay_steps=2),
    Condition("obs_noise_train_level", obs_noise=True),
]
COND_BY_NAME = {c.name: c for c in CONDITIONS}


def summarize(eps: list[dict]) -> dict:
    """Fall rate with Wilson 95% interval, tracking error and survival time with normal 95% CIs."""
    n = len(eps)
    falls = sum(e["fell"] for e in eps)
    lo, hi = wilson(falls, n)
    lin = [e["lin_vel_err"] for e in eps if e["lin_vel_err"] is not None]
    yaw = [e["yaw_rate_err"] for e in eps if e["yaw_rate_err"] is not None]
    surv = [e["survival_s"] for e in eps]
    m_lin, lin_lo, lin_hi = mean_ci(lin)
    m_yaw, yaw_lo, yaw_hi = mean_ci(yaw)
    m_s, _, _ = mean_ci(surv)
    return {
        "episodes": n,
        "falls": falls,
        "fall_rate": falls / n if n else None,
        "fall_rate_wilson95": [lo, hi],
        "lin_vel_err_mean": m_lin,
        "lin_vel_err_ci95": [lin_lo, lin_hi],
        "yaw_rate_err_mean": m_yaw,
        "yaw_rate_err_ci95": [yaw_lo, yaw_hi],
        "survival_s_mean": m_s,
        "ended_low_count": sum(e.get("ended_low", False) for e in eps),
    }


def run_condition(args: argparse.Namespace) -> None:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, device=args.device).app
    try:
        _run_condition(args)
    finally:
        app.close()


def _run_condition(args: argparse.Namespace) -> None:
    from collections import deque

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.locomotion.velocity import mdp
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cond = COND_BY_NAME[args.condition]
    n = args.episodes
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n)
    cfg.seed = BASE_SEED
    cfg.observations.policy.enable_corruption = cond.obs_noise
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    c = cfg.commands.base_velocity
    c.heading_command = False
    c.rel_heading_envs = 0.0
    c.rel_standing_envs = 0.0
    c.resampling_time_range = (1.0e6, 1.0e6)
    ranges = {k: list(getattr(c.ranges, k)) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")}  # task training ranges
    c.debug_vis = False
    cfg.episode_length_s = EPISODE_STEPS * 0.02
    pm = cfg.events.physics_material.params
    nominal_friction = {"static": pm["static_friction_range"][0], "dynamic": pm["dynamic_friction_range"][0]}
    s = cond.friction_scale
    pm["static_friction_range"] = (nominal_friction["static"] * s,) * 2
    pm["dynamic_friction_range"] = (nominal_friction["dynamic"] * s,) * 2
    if cond.torso_mass_delta_kg:
        d = cond.torso_mass_delta_kg
        cfg.events.add_base_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                    "mass_distribution_params": (d, d), "operation": "add"})

    env = gym.make(args.task, cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    uenv = env.unwrapped
    dev = uenv.device
    robot = uenv.scene["robot"]
    torso_id = robot.find_bodies("torso_link")[0][0]
    torso_mass = float(robot.root_physx_view.get_masses()[0, torso_id])
    policy = torch.jit.load(str(ROOT / args.policy), map_location=dev).eval()

    g = torch.Generator(device="cpu").manual_seed(BASE_SEED + 7_000_000)
    next_push = torch.round((torch.rand(n, generator=g) * (cond.push_interval_s[1] - cond.push_interval_s[0])
                             + cond.push_interval_s[0]) / 0.02).long().to(dev)
    obs, _ = env.get_observations()
    delay = deque([torch.zeros(n, env.num_actions, device=dev)] * cond.action_delay_steps)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    t_end = torch.full((n,), EPISODE_STEPS, device=dev)
    lin_sum = torch.zeros(n, device=dev)
    yaw_sum = torch.zeros(n, device=dev)
    cnt = torch.zeros(n, device=dev)
    n_push = torch.zeros(n, dtype=torch.long, device=dev)
    cmd0 = uenv.command_manager.get_command("base_velocity").clone()
    t0 = time.time()
    for t in range(EPISODE_STEPS):
        with torch.inference_mode():
            act = policy(obs)
        if cond.push:
            ids = ((next_push == t) & alive).nonzero(as_tuple=False).flatten()
            if len(ids):
                k = len(ids)
                theta = torch.rand(k, generator=g).to(dev) * 2 * torch.pi
                mag = (torch.rand(k, generator=g) * (cond.push_magnitude[1] - cond.push_magnitude[0])
                       + cond.push_magnitude[0]).to(dev)
                vel = robot.data.root_vel_w[ids].clone()
                vel[:, 0] += mag * torch.cos(theta)
                vel[:, 1] += mag * torch.sin(theta)
                robot.write_root_velocity_to_sim(vel, env_ids=ids)
                n_push[ids] += 1
                gap = torch.round((torch.rand(k, generator=g) * (cond.push_interval_s[1] - cond.push_interval_s[0])
                                   + cond.push_interval_s[0]) / 0.02).long().to(dev)
                next_push[ids] = t + gap
        delay.append(act)
        applied = delay.popleft()
        obs, _, dones, _ = env.step(applied)
        term = uenv.reset_terminated.clone()
        new_fall = alive & term
        fell |= new_fall
        t_end[new_fall] = t + 1
        ended = alive & (dones > 0)
        still = alive & ~ended
        if t >= WARMUP_STEPS:
            cmd = uenv.command_manager.get_command("base_velocity")
            v = robot.data.root_lin_vel_b
            w = robot.data.root_ang_vel_b
            le = torch.linalg.norm(cmd[:, :2] - v[:, :2], dim=1)
            ye = (cmd[:, 2] - w[:, 2]).abs()
            lin_sum += torch.where(still, le, torch.zeros_like(le))
            yaw_sum += torch.where(still, ye, torch.zeros_like(ye))
            cnt += still.float()
        if t == EPISODE_STEPS - 2:
            # base height one step before the time-out reset, for robots that did not fall
            low = robot.data.root_pos_w[:, 2] < 0.5
        alive &= ~ended
    wall = time.time() - t0
    eps = []
    for i in range(n):
        c_i = float(cnt[i])
        eps.append({
            "env": i,
            "command": [float(x) for x in cmd0[i].tolist()],
            "fell": bool(fell[i]),
            "survival_s": float(t_end[i]) * 0.02,
            "lin_vel_err": float(lin_sum[i] / c_i) if c_i > 0 else None,
            "yaw_rate_err": float(yaw_sum[i] / c_i) if c_i > 0 else None,
            "pushes": int(n_push[i]),
            "ended_low": bool(low[i]) and not bool(fell[i]),
        })
    out = {
        "condition": asdict(cond),
        "task": args.task,
        "policy": args.policy,
        "command_ranges": ranges,
        "robot_friction_static_dynamic": [nominal_friction["static"] * s, nominal_friction["dynamic"] * s],
        "torso_link_mass_kg_env0": torso_mass,
        "rollout_wall_s": wall,
        "summary": summarize(eps),
        "episodes": eps,
    }
    Path(args.part_out).write_text(json.dumps(out))
    env.close()


def run_all(args: argparse.Namespace) -> None:
    from hloco.common import RESULTS, write_json

    part_dir = ROOT / "logs" / "isaac" / "eval_parts" / args.name
    part_dir.mkdir(parents=True, exist_ok=True)
    conds = args.conditions or [c.name for c in CONDITIONS]
    results, failures = {}, {}
    t0 = time.time()
    for cname in conds:
        part = part_dir / f"{cname}.json"
        log = ROOT / "logs" / "isaac" / f"eval_{args.name}_{cname}.log"
        cmd = [sys.executable, __file__, "--condition", cname, "--episodes", str(args.episodes), "--task", args.task,
               "--policy", args.policy, "--part-out", str(part), "--device", args.device]
        if part.exists():
            part.unlink()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=os.environ.copy(), timeout=3600).returncode
        if part.exists():
            results[cname] = json.loads(part.read_text())
            s = results[cname]["summary"]
            print(f"{cname}: falls {s['falls']}/{s['episodes']} lin err {s['lin_vel_err_mean']:.3f}", flush=True)
        else:
            failures[cname] = f"subprocess exit code {rc}, no result (see {log.relative_to(ROOT)})"
            print(f"{cname}: FAILED rc={rc}", flush=True)
    write_json(RESULTS / f"isaac_eval_{args.name}.json", {
        "label": "within Isaac Sim (PhysX), not sim-to-sim",
        "task": args.task,
        "policy": args.policy,
        "episodes_per_condition": args.episodes,
        "episode_steps": EPISODE_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "protocol": __doc__.split("Protocol", 1)[1].strip(),
        "wall_clock_s": time.time() - t0,
        "summary": {k: v["summary"] for k, v in results.items()},
        "details": {k: {kk: vv for kk, vv in v.items() if kk not in ("summary", "episodes")} for k, v in results.items()},
        "episodes": {k: v["episodes"] for k, v in results.items()},
        "failed_conditions": failures,
        "status": "completed" if not failures else f"failed conditions: {sorted(failures)}",
    })
    print("wrote", RESULTS / f"isaac_eval_{args.name}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--name", default="g1_flat")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--condition", default="nominal")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0")
    ap.add_argument("--policy", default="checkpoints/isaac_g1_flat/policy.pt")
    ap.add_argument("--part-out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.all:
        run_all(args)
    else:
        run_condition(args)


if __name__ == "__main__":
    main()
