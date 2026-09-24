"""Train G1JoystickFlatTerrain with rsl-rl-lib PPO through Playground's torch wrapper.

Follows mujoco_playground's learning/train_rsl_rl.py: RSLRLBraxWrapper (MJX env on GPU,
torch tensors via DLPack), locomotion_params.rsl_rl_config, asymmetric actor/critic
observations, and the env's domain randomizer (the upstream script always enables it).

Usage: python scripts/train_rsl_rl.py --name rsl_dr --cap-minutes 75
Writes results/train_<name>.json, checkpoints/<name>/model.pt and policy.pkl (numpy actor).
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")
os.environ.setdefault("MUJOCO_GL", "egl")

from hloco.common import CKPT, ENV_NAME, RESULTS, ROOT, write_json  # noqa: E402


class WallClockCap(Exception):
    pass


def export_actor(runner) -> dict:
    """Numpy copy of the deterministic actor: (obs - mean) / (std + eps) -> MLP(elu) -> clip."""
    import numpy as np

    pol = runner.alg.policy
    norm = pol.actor_obs_normalizer
    mean = norm._mean.squeeze(0).detach().cpu().numpy()
    std = (norm._std.squeeze(0) + norm.eps).detach().cpu().numpy()
    linears = [m for m in pol.actor.modules() if m.__class__.__name__ == "Linear"]
    layers = [{"w": m.weight.detach().cpu().numpy().T.copy(), "b": m.bias.detach().cpu().numpy().copy()}
              for m in linears]
    acts = {m.__class__.__name__ for m in pol.actor.modules()} & {"ELU", "SiLU", "ReLU", "Tanh"}
    assert acts == {"ELU"}, acts
    return {"obs_key": "state", "mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "layers": layers, "activation": "elu", "squash": "clip"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="rsl_dr")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap-minutes", type=float, default=75.0)
    ap.add_argument("--max-iterations", type=int, default=100000)
    ap.add_argument("--no-dr", action="store_true")
    args = ap.parse_args()

    import jax
    import torch
    from mujoco_playground import registry, wrapper_torch
    from mujoco_playground.config import locomotion_params
    from rsl_rl.runners import OnPolicyRunner

    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    torch.manual_seed(args.seed)

    env_cfg = registry.get_default_config(ENV_NAME)
    raw_env = registry.load(ENV_NAME, config=env_cfg)
    randomizer = None if args.no_dr else registry.get_domain_randomizer(ENV_NAME)
    env = wrapper_torch.RSLRLBraxWrapper(
        raw_env, args.num_envs, args.seed, env_cfg.episode_length, 1,
        randomization_fn=randomizer, device_rank=0,
    )
    # rsl-rl-lib 3.3 reads env.cfg for logging; Playground's wrapper does not define it.
    env.cfg = env_cfg.to_dict()
    train_cfg = locomotion_params.rsl_rl_config(ENV_NAME)
    train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]}
    train_cfg.seed = args.seed
    train_cfg.max_iterations = args.max_iterations
    train_cfg.experiment_name = args.name
    train_cfg.save_interval = 10**9  # checkpointing handled below
    out_dir = CKPT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / f"tb_{args.name}"
    runner = OnPolicyRunner(env, train_cfg.to_dict(), str(log_dir), device="cuda:0")

    curve: list[dict] = []
    t0 = time.monotonic()
    state = {"stopped_by_cap": False, "first_iter_wall": None}
    steps_per_iter = args.num_envs * train_cfg.num_steps_per_env

    def dump(status: str) -> None:
        sps = None
        if len(curve) >= 2:
            dt = curve[-1]["wall_s"] - curve[0]["wall_s"]
            sps = (curve[-1]["step"] - curve[0]["step"]) / dt if dt > 0 else None
        write_json(RESULTS / f"train_{args.name}.json", {
            "name": args.name,
            "env": ENV_NAME,
            "algo": "rsl-rl-lib PPO (OnPolicyRunner) via mujoco_playground wrapper_torch.RSLRLBraxWrapper",
            "domain_randomization": not args.no_dr,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "status": status,
            "stopped_by_wall_clock_cap": state["stopped_by_cap"],
            "cap_minutes": args.cap_minutes,
            "final_step": curve[-1]["step"] if curve else 0,
            "iterations": len(curve),
            "wall_clock_s": time.monotonic() - t0,
            "env_steps_per_s_steady": sps,
            "rsl_rl_config": train_cfg.to_dict(),
            "reward_definition": "mean undiscounted training-episode return over the last 100 finished episodes (stochastic policy, DR on)",
            "curve": curve[:: max(1, len(curve) // 400)] + ([curve[-1]] if curve else []),
            "checkpoint": str((out_dir / "model.pt").relative_to(ROOT)),
        })

    orig_log = runner.logger.log

    def log_hook(*a, **kw):
        orig_log(*a, **kw)
        lg = runner.logger
        it = kw.get("it", a[0] if a else None)
        now = time.monotonic() - t0
        rec = {"iter": int(it), "step": int((it + 1) * steps_per_iter), "wall_s": now}
        if len(lg.rewbuffer) > 0:
            rec["mean_episode_reward"] = float(statistics.mean(lg.rewbuffer))
            rec["mean_episode_length"] = float(statistics.mean(lg.lenbuffer))
        curve.append(rec)
        if it % 25 == 0:
            print(f"[{now:8.1f}s] it={it} step={rec['step']} reward={rec.get('mean_episode_reward')} "
                  f"len={rec.get('mean_episode_length')}", flush=True)
            dump("running")
        if it % 100 == 0:
            runner.save(str(out_dir / "model.pt"))
        if now > args.cap_minutes * 60:
            state["stopped_by_cap"] = True
            raise WallClockCap()

    runner.logger.log = log_hook
    status = "finished"
    try:
        runner.learn(num_learning_iterations=train_cfg.max_iterations, init_at_random_ep_len=False)
    except WallClockCap:
        status = "stopped_by_wall_clock_cap"
    except Exception as e:  # noqa: BLE001
        status = f"failed: {type(e).__name__}: {e}"
        dump(status)
        raise
    runner.save(str(out_dir / "model.pt"))
    actor = export_actor(runner)
    with open(out_dir / "policy.pkl", "wb") as f:
        pickle.dump({"policy": actor, "extra": {"iter": curve[-1]["iter"] if curve else 0,
                                                "step": curve[-1]["step"] if curve else 0}}, f)
    dump(status)
    print("done", status)


if __name__ == "__main__":
    main()
