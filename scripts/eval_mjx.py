"""Common in-simulator (MJX) evaluation for every exported actor, brax or RSL-RL.

The actor is rebuilt in JAX from the exported numpy weights (the same weights the ONNX file
holds), so brax PPO and RSL-RL PPO policies are scored by one identical procedure:
N parallel envs, one 1000-step episode each (20 s), deterministic actions, env reward.
Two settings: training env with its domain randomizer, and the nominal env (no DR).

Usage: python scripts/eval_mjx.py --policy brax_dr [--envs 1024]
Writes results/mjx_eval_<policy>.json.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import numpy as np  # noqa: E402

from hloco.common import ENV_NAME, RESULTS, ROOT, write_json  # noqa: E402
from hloco.policy import jax_actor, load_actor  # noqa: E402
from hloco.stats import mean_ci, wilson  # noqa: E402

EVAL_SEED = 2024


def evaluate(policy: dict, n_envs: int, dr: bool) -> dict:
    import jax
    import jax.numpy as jp
    from mujoco_playground import registry, wrapper

    env = registry.load(ENV_NAME)
    key = jax.random.PRNGKey(EVAL_SEED)
    k_rand, k_reset = jax.random.split(key)
    rand_fn = None
    if dr:
        import functools

        rand_fn = functools.partial(registry.get_domain_randomizer(ENV_NAME),
                                    rng=jax.random.split(k_rand, n_envs))
    wenv = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1, randomization_fn=rand_fn)
    actor = jax_actor(policy)
    steps = 1000

    @jax.jit
    def run(k):
        s = wenv.reset(jax.random.split(k, n_envs))

        def body(carry, _):
            s, alive, ret, length = carry
            a = actor(s.obs["state"])
            s = wenv.step(s, a)
            ret = ret + s.reward * alive
            length = length + alive
            terminated = (s.done > 0) & (s.info["truncation"] == 0)
            alive = alive * (1.0 - terminated.astype(jp.float32))
            return (s, alive, ret, length), None

        init = (s, jp.ones(n_envs), jp.zeros(n_envs), jp.zeros(n_envs))
        (s, alive, ret, length), _ = jax.lax.scan(body, init, None, length=steps)
        return alive, ret, length

    alive, ret, length = jax.device_get(run(k_reset))
    falls = int((alive < 0.5).sum())
    lo, hi = wilson(falls, n_envs)
    m, rlo, rhi = mean_ci([float(x) for x in ret])
    return {
        "envs": n_envs,
        "episode_steps": steps,
        "mean_return": m,
        "return_ci95": [rlo, rhi],
        "falls": falls,
        "fall_rate": falls / n_envs,
        "fall_rate_wilson95": [lo, hi],
        "mean_episode_length_steps": float(np.mean(length)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--envs", type=int, default=1024)
    args = ap.parse_args()
    # Workaround: in this venv (torch 2.8 pins the nvidia cu12.8 wheels that JAX also uses) a fresh
    # process that imports JAX alone fails every GPU matmul with "INTERNAL: an unsupported value or
    # parameter was passed to the function". Importing torch first loads its CUDA libraries and the
    # JAX matmuls then run. Observed and worked around, root cause not investigated further.
    import torch

    torch.cuda.is_available()  # load torch's CUDA libraries before JAX initializes its GPU backend
    import jax

    jax.config.update("jax_compilation_cache_dir", str(ROOT / ".jax_cache"))
    actor = load_actor(args.policy)
    out = {
        "policy": args.policy,
        "checkpoint_step": actor["step"],
        "device": str(jax.devices()[0]),
        "eval_seed": EVAL_SEED,
        "protocol": "MJX training env (obs noise and pushes as in training), deterministic actor, "
                    "return = sum of env reward until termination or 1000 steps; commands resampled by the env",
        "with_dr": evaluate(actor["policy"], args.envs, dr=True),
        "without_dr": evaluate(actor["policy"], args.envs, dr=False),
    }
    write_json(RESULTS / f"mjx_eval_{args.policy}.json", out)
    print({k: out[k]["mean_return"] for k in ("with_dr", "without_dr")})


if __name__ == "__main__":
    main()
