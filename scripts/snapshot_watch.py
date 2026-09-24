"""Keep a copy of every params.pkl a running brax job writes (one per evaluation point).

Usage: python scripts/snapshot_watch.py --run brax_dr
Copies checkpoints/<run>/params.pkl to checkpoints/<run>/snapshots/params_step<N>.pkl whenever
the step stored inside the file changes. Used for step-matched DR vs no-DR comparisons.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # unpickling brax params imports jax; keep it off the GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hloco.common import CKPT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--max-hours", type=float, default=4.0)
    args = ap.parse_args()
    src = CKPT / args.run / "params.pkl"
    dst = CKPT / args.run / "snapshots"
    dst.mkdir(parents=True, exist_ok=True)
    seen = {int(p.stem.split("step")[-1]) for p in dst.glob("params_step*.pkl")}
    t_end = time.time() + args.max_hours * 3600
    while time.time() < t_end:
        if src.exists():
            try:
                with open(src, "rb") as f:
                    step = int(pickle.load(f)["extra"]["step"])
                if step not in seen:
                    shutil.copy2(src, dst / f"params_step{step}.pkl")
                    seen.add(step)
                    print("snapshot", step, flush=True)
            except Exception as e:  # noqa: BLE001  (file being replaced)
                print("retry:", type(e).__name__, flush=True)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    main()
