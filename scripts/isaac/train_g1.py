"""Train an Isaac Lab velocity-tracking task (default: Isaac-Velocity-Flat-G1-v0) with RSL-RL PPO, headless.

Runs inside the Isaac Sim / Isaac Lab venv (.venv-isaac, see scripts/isaac/setup_isaac.sh), not the main venv.
Uses the task's registered env config and its registered RSL-RL runner config unchanged, except for
num_envs, seed and a wall-clock cap. The per-iteration curve is written to results/isaac_train_<name>.json
while training runs (every 10 iterations) and once more at the end.

Example:
    OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/isaac/train_g1.py --headless \
        --task Isaac-Velocity-Flat-G1-v0 --name g1_flat --num_envs 4096 --cap-minutes 115
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-v0")
parser.add_argument("--name", default="g1_flat", help="results/isaac_train_<name>.json, logs/isaac/runs/<name>")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None, help="default: the task's runner config")
parser.add_argument("--cap-minutes", type=float, default=115.0, help="stop after the iteration that crosses this")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hloco.common import RESULTS, write_json  # noqa: E402


class StopTraining(Exception):
    pass


class CappedRunner(OnPolicyRunner):
    """OnPolicyRunner that records a per-iteration curve and stops at a wall-clock cap."""

    def setup_capture(self, t0: float, cap_s: float, out: Path, static: dict) -> None:
        self.t0, self.cap_s, self.out, self.static = t0, cap_s, out, static
        self.curve: list[dict] = []
        self.stopped_by_cap = False

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        super().log(locs, width, pad)
        it = locs["it"]
        rec = {
            "iteration": it,
            "step": self.tot_timesteps,
            "wall_s": round(time.time() - self.t0, 2),
            "collection_s": round(locs["collection_time"], 4),
            "learn_s": round(locs["learn_time"], 4),
            "fps": round(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]), 1),
            "action_noise_std": float(self.alg.policy.action_std.mean().item()),
        }
        if len(locs["rewbuffer"]) > 0:
            rec["mean_episode_reward"] = statistics.mean(locs["rewbuffer"])
            rec["mean_episode_length"] = statistics.mean(locs["lenbuffer"])
        for key in ("Episode_Termination/base_contact", "Episode_Termination/time_out",
                    "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw"):
            vals = [float(torch.as_tensor(e[key]).float().mean()) for e in locs["ep_infos"] if key in e]
            if vals:
                rec[key] = sum(vals) / len(vals)
        self.curve.append(rec)
        if it % 10 == 0:
            self.dump("running")
        if time.time() - self.t0 > self.cap_s:
            self.stopped_by_cap = True
            self.save(str(Path(self.log_dir) / f"model_{it}.pt"))
            raise StopTraining

    def dump(self, status: str) -> None:
        c = self.curve
        half = c[len(c) // 2:] if len(c) >= 4 else c
        payload = {
            **self.static,
            "status": status,
            "iterations_completed": len(c),
            "final_iteration": c[-1]["iteration"] if c else None,
            "final_step": c[-1]["step"] if c else 0,
            "wall_clock_s": round(time.time() - self.t0, 1),
            "train_wall_s_from_first_iteration": c[-1]["wall_s"] - c[0]["wall_s"] + c[0]["collection_s"] + c[0]["learn_s"] if c else None,
            "env_steps_per_s_steady": statistics.median(r["fps"] for r in half) if half else None,
            "curve": c,
        }
        write_json(self.out, payload)


def main() -> None:
    t_start = time.time()
    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    env_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    run_dir = ROOT / "logs" / "isaac" / "runs" / args.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)
    dump_pickle(str(run_dir / "params" / "env.pkl"), env_cfg)
    dump_pickle(str(run_dir / "params" / "agent.pkl"), agent_cfg)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = CappedRunner(env, agent_cfg.to_dict(), log_dir=str(run_dir), device=agent_cfg.device)

    import importlib.metadata as md

    static = {
        "task": args.task,
        "framework": f"rsl-rl-lib {md.version('rsl-rl-lib')} PPO (Isaac Lab registered runner config)",
        "isaac_sim": md.version("isaacsim"),
        "isaac_lab": md.version("isaaclab"),
        "isaaclab_tasks": md.version("isaaclab_tasks"),
        "torch": torch.__version__,
        "simulator": "Isaac Sim (PhysX, GPU)",
        "num_envs": args.num_envs,
        "seed": args.seed,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "requested_max_iterations": agent_cfg.max_iterations,
        "cap_minutes": args.cap_minutes,
        "policy_dt_s": env.unwrapped.step_dt,
        "physics_dt_s": env_cfg.sim.dt,
        "episode_length_s": env_cfg.episode_length_s,
        "obs_dim": env.num_obs,
        "action_dim": env.num_actions,
        "actor_hidden_dims": list(agent_cfg.policy.actor_hidden_dims),
        "domain_randomization_in_training": "task defaults: robot link friction fixed static 0.8 / dynamic 0.6, "
        "reset pose/yaw randomization, reset joint scale (1.0, 1.0), observation noise; "
        "the G1 rough/flat configs disable push_robot, add_base_mass and base_com",
        "startup_s": round(time.time() - t_start, 1),
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    out = RESULTS / f"isaac_train_{args.name}.json"
    runner.setup_capture(time.time(), args.cap_minutes * 60, out, static)
    status = "completed"
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except StopTraining:
        status = "stopped at wall-clock cap"
    except Exception as e:  # noqa: BLE001
        status = f"failed: {type(e).__name__}: {e}"
        runner.dump(status)
        raise
    if runner.stopped_by_cap:
        status = "stopped at wall-clock cap"
    runner.dump(status)
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    print(json.dumps({"status": status, "final_checkpoint": str(ckpts[-1]) if ckpts else None}))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
