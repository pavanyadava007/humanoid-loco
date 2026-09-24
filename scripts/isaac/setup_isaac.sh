#!/usr/bin/env bash
# Install NVIDIA Isaac Sim 4.5.0 (pip wheels) and Isaac Lab v2.1.1 into a separate venv, .venv-isaac.
# Installing Isaac Sim means accepting the NVIDIA Omniverse EULA; every Isaac command in this repo sets
# OMNI_KIT_ACCEPT_EULA=YES. Measured on Amazon Linux 2023 (glibc 2.34), Python 3.10, NVIDIA L4, driver 580.95.05.
set -euo pipefail
cd "$(dirname "$0")/../.."
UV=${UV:-uv}
export UV_HTTP_TIMEOUT=600
[ -f /etc/pki/tls/certs/ca-bundle.crt ] && export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt

$UV venv --python 3.10 .venv-isaac
PY=.venv-isaac/bin/python

# 1. PyTorch 2.5.1 + CUDA 12.1 (the combination the Isaac Sim 4.5 pip docs use)
$UV pip install --python $PY torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# 2. Isaac Sim 4.5.0 with extension caches (about 4.3 GB of wheels from pypi.nvidia.com)
$UV pip install --python $PY "isaacsim[all,extscache]==4.5.0.0" --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match
# 3. Isaac Lab v2.1.1 (supports Isaac Sim 4.5). isaaclab.sh -i would replace torch with 2.7.0+cu128; keep 2.5.1.
if [ ! -d third_party/IsaacLab ]; then
  git clone --depth 1 --branch v2.1.1 https://github.com/isaac-sim/IsaacLab.git third_party/IsaacLab
fi
# flatdict 4.0.1 (pinned by isaaclab) imports pkg_resources in setup.py: build it without isolation
$UV pip install --python $PY "setuptools<80" wheel
$UV pip install --python $PY --no-build-isolation flatdict==4.0.1
printf 'torch==2.5.1+cu121\ntorchvision==0.20.1+cu121\n' > /tmp/isaac_constraints.txt
L=third_party/IsaacLab/source
$UV pip install --python $PY -c /tmp/isaac_constraints.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  -e $L/isaaclab -e $L/isaaclab_assets -e $L/isaaclab_tasks -e "$L/isaaclab_rl[rsl_rl]"
$UV pip install --python $PY onnxruntime==1.19.2
# 4. Camera rendering (video) needs libGLU.so.1, missing on a minimal Amazon Linux 2023 install:
#    sudo dnf install -y mesa-libGLU
echo "Isaac venv ready: $PY"
