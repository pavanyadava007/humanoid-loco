"""Record what the Isaac Sim / Isaac Lab install took on this host: versions, disk, time, smoke test.

Reads the install and smoke-test logs written by the commands in docs/ISAAC_LAB.md (logs/isaac/*.log),
measures disk use with du and queries package versions from .venv-isaac. Writes results/isaac_install.json.
Stdlib only; run with either venv.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hloco.common import RESULTS, ROOT, write_json  # noqa: E402

LOGS = ROOT / "logs" / "isaac"
DATE_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\w{3}) +(\d+) (\d\d:\d\d:\d\d) (AM|PM) UTC (\d{4})$", re.M)


def stamps(text: str) -> list[dt.datetime]:
    out = []
    for m in DATE_RE.finditer(text):
        out.append(dt.datetime.strptime(f"{m[2]} {m[3]} {m[6]} {m[4]} {m[5]}", "%b %d %Y %I:%M:%S %p"))
    return out


def du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def versions() -> dict:
    py = ROOT / ".venv-isaac" / "bin" / "python"
    code = ("import importlib.metadata as m, json;"
            "print(json.dumps({p: m.version(p) for p in ['isaacsim','isaaclab','isaaclab_tasks','isaaclab_rl',"
            "'isaaclab_assets','rsl-rl-lib','torch','numpy','gymnasium','warp-lang','onnx','onnxruntime']}))")
    import json

    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-400:]}


def main() -> None:
    sim_log = (LOGS / "install_sim.log").read_text()
    lab_log = (LOGS / "install_lab.log").read_text()
    smoke = (LOGS / "smoke_cartpole.log").read_text()
    ts = stamps(sim_log)
    tl = stamps(lab_log)
    free = [ln for ln in sim_log.splitlines() if ln.startswith("/dev/")]
    graphics = re.search(r"Graphics API: (\w+)", smoke)
    fps = re.findall(r"Computation: (\d+) steps/s", smoke)
    exit_code = re.search(r"^EXIT (\d+)", smoke, re.M)
    lab_git = subprocess.run(["git", "-C", str(ROOT / "third_party" / "IsaacLab"), "describe", "--tags"],
                             capture_output=True, text=True).stdout.strip()
    write_json(RESULTS / "isaac_install.json", {
        "host_os": Path("/etc/system-release").read_text().strip() if Path("/etc/system-release").exists() else None,
        "glibc": subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0].split()[-1],
        "venv": ".venv-isaac (Python 3.10)",
        "versions": versions(),
        "isaac_lab_git_tag": lab_git,
        "install_torch_s": (ts[1] - ts[0]).total_seconds() if len(ts) >= 2 else None,
        "install_isaacsim_s": (ts[2] - ts[1]).total_seconds() if len(ts) >= 3 else None,
        "install_isaaclab_s": (tl[-1] - tl[0]).total_seconds() if len(tl) >= 2 else None,
        "df_lines_during_isaacsim_install": free,
        "disk_bytes": {
            ".venv-isaac": du_bytes(ROOT / ".venv-isaac"),
            "third_party/IsaacLab": du_bytes(ROOT / "third_party" / "IsaacLab"),
            "logs/isaac": du_bytes(LOGS),
            "~/.cache/ov (Omniverse runtime cache: downloaded assets, shaders)": du_bytes(Path.home() / ".cache" / "ov"),
            "~/.cache/uv (hard-links the same files as .venv-isaac, not extra disk)": du_bytes(Path.home() / ".cache" / "uv"),
        },
        "note_disk": "du -sb counts apparent size per path; .venv-isaac and the uv cache share hard-linked files, "
                     "so their sum overstates real use; the filesystem used-space readings during the install are listed below.",
        "extra_system_package": "mesa-libGLU (dnf), needed by the RTX renderer for camera/video rendering; "
        "without it the --enable_cameras run segfaulted after 'libGLU.so.1: cannot open shared object file'",
        "smoke_test": {
            "command": "scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Cartpole-v0 --headless "
                       "--num_envs 1024 --max_iterations 5",
            "exit_code": int(exit_code[1]) if exit_code else None,
            "graphics_api": graphics[1] if graphics else None,
            "iterations_logged": len(fps),
            "last_iteration_steps_per_s": int(fps[-1]) if fps else None,
        },
        "eula": "Isaac Sim requires accepting the NVIDIA Omniverse EULA; accepted by the repository owner via "
                "OMNI_KIT_ACCEPT_EULA=YES",
    })
    print("wrote", RESULTS / "isaac_install.json")


if __name__ == "__main__":
    main()
