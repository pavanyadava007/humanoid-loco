#!/usr/bin/env bash
# Reproducible environment: Python 3.10 venv, pinned JAX/MuJoCo/Playground/torch/rsl-rl,
# and a sparse checkout of the one MuJoCo Menagerie model Playground needs (unitree_g1).
set -euo pipefail
cd "$(dirname "$0")/.."
UV=${UV:-uv}
$UV venv -p 3.10 .venv
$UV pip install -p .venv/bin/python -r requirements.txt
MENAGERIE_SHA=1b86ece576591213e2b666ebf59508454200ca97  # pinned in mujoco_playground/_src/mjx_env.py
EXT=.venv/lib/python3.10/site-packages/mujoco_playground/external_deps
if [ ! -d "$EXT/mujoco_menagerie/unitree_g1" ]; then
  mkdir -p "$EXT"
  git clone -q --filter=blob:none --no-checkout https://github.com/deepmind/mujoco_menagerie.git "$EXT/mujoco_menagerie"
  git -C "$EXT/mujoco_menagerie" sparse-checkout set unitree_g1
  git -C "$EXT/mujoco_menagerie" checkout -q "$MENAGERIE_SHA"
fi
.venv/bin/python -c "import jax; print('jax devices:', jax.devices())"
