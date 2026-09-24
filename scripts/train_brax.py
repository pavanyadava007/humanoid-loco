"""Train G1JoystickFlatTerrain with brax PPO (Playground default locomotion config).

Usage:
  python scripts/train_brax.py --name brax_dr --dr --num-timesteps 100000000
  python scripts/train_brax.py --name brax_nodr --no-dr --num-timesteps 100000000

Writes results/train_<name>.json (curve + throughput) incrementally and
checkpoints/<name>/params.pkl at every evaluation point. A wall-clock cap
stops training at the first evaluation after the cap and keeps the last params.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax  # noqa: E402

from hloco import brax_params  # noqa: E402
from hloco.common import CKPT, ENV_NAME, RESULTS, ROOT, write_json  # noqa: E402


class WallClockCap(Exception):
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dr", dest="dr", action="store_true")
    ap.add_argument("--no-dr", dest="dr", action="store_false")
    ap.set_defaults(dr=True)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--num-evals", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap-minutes", type=float, default=125.0)
    args = ap.parse_args()

    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    env_cfg = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=env_cfg)
    eval_env = registry.load(ENV_NAME, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    default_timesteps = int(ppo_params.num_timesteps)
    if args.num_timesteps is not None:
        ppo_params.num_timesteps = args.num_timesteps
    if args.num_evals is not None:
        ppo_params.num_evals = args.num_evals
    if args.num_envs is not None:
        ppo_params.num_envs = args.num_envs

    training_params = dict(ppo_params)
    net_cfg = training_params.pop("network_factory")
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)
    if args.dr:
        training_params["randomization_fn"] = registry.get_domain_randomizer(ENV_NAME)

    out_json = RESULTS / f"train_{args.name}.json"
    ckpt_dir = CKPT / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curve: list[dict] = []
    t0 = time.monotonic()
    state = {"stopped_by_cap": False, "last_saved_step": None}

    def dump(status: str) -> None:
        train_pts = [p for p in curve if p["step"] > 0]
        sps = None
        if len(train_pts) >= 1 and curve:
            # Throughput between the first (step 0) eval and the last eval; includes in-loop evals
            # and the first-iteration JIT compile of the training step.
            dt = curve[-1]["wall_s"] - curve[0]["wall_s"]
            sps = (curve[-1]["step"] - curve[0]["step"]) / dt if dt > 0 else None
        sps_steady = None
        if len(train_pts) >= 2:
            dt = train_pts[-1]["wall_s"] - train_pts[0]["wall_s"]
            sps_steady = (train_pts[-1]["step"] - train_pts[0]["step"]) / dt if dt > 0 else None
        write_json(
            out_json,
            {
                "name": args.name,
                "env": ENV_NAME,
                "algo": "brax PPO (mujoco_playground locomotion_params.brax_ppo_config)",
                "domain_randomization": args.dr,
                "seed": args.seed,
                "status": status,
                "stopped_by_wall_clock_cap": state["stopped_by_cap"],
                "cap_minutes": args.cap_minutes,
                "default_num_timesteps_in_config": default_timesteps,
                "requested_num_timesteps": int(ppo_params.num_timesteps),
                "final_step": curve[-1]["step"] if curve else 0,
                "wall_clock_s": time.monotonic() - t0,
                "env_steps_per_s_incl_evals_and_compile": sps,
                "env_steps_per_s_steady": sps_steady,
                "ppo_config": {k: (v if not hasattr(v, "to_dict") else v.to_dict()) for k, v in ppo_params.items()},
                "env_config": env_cfg.to_dict(),
                "curve": curve,
                "checkpoint": str((ckpt_dir / "params.pkl").relative_to(ROOT)),
            },
        )

    def progress(num_steps, metrics):
        rec = {"step": int(num_steps), "wall_s": time.monotonic() - t0}
        for k in ("eval/episode_reward", "eval/episode_reward_std", "eval/avg_episode_length"):
            if k in metrics:
                rec[k.split("/")[-1]] = float(metrics[k])
        for k, v in metrics.items():
            if k.startswith("eval/episode_reward/"):
                rec.setdefault("reward_terms", {})[k.split("/")[-1]] = float(v)
        curve.append(rec)
        print(
            f"[{rec['wall_s']:8.1f}s] step={num_steps:>11d} "
            f"reward={rec.get('episode_reward', float('nan')):.3f} "
            f"len={rec.get('avg_episode_length', float('nan')):.1f}",
            flush=True,
        )
        dump("running")
        # params for this eval were saved by policy_params_fn just before the eval ran
        if rec["wall_s"] > args.cap_minutes * 60 and num_steps > 0:
            state["stopped_by_cap"] = True
            raise WallClockCap()

    def policy_params_fn(current_step, make_policy, params):
        del make_policy
        brax_params.save(
            ckpt_dir / "params.pkl",
            params,
            extra={"step": int(current_step), "dr": args.dr, "seed": args.seed},
        )
        state["last_saved_step"] = int(current_step)

    try:
        ppo.train(
            environment=env,
            eval_env=eval_env,
            **training_params,
            network_factory=network_factory,
            seed=args.seed,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            progress_fn=progress,
            policy_params_fn=policy_params_fn,
        )
        dump("finished")
    except WallClockCap:
        dump("stopped_by_wall_clock_cap")
    except Exception as e:  # noqa: BLE001
        dump(f"failed: {type(e).__name__}: {e}")
        raise
    print("done", out_json)


if __name__ == "__main__":
    main()
