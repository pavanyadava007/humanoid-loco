"""Shared paths, labels and JSON helpers."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MEDIA = ROOT / "media"
CKPT = ROOT / "checkpoints"

ENV_NAME = "G1JoystickFlatTerrain"
HARDWARE_LABEL = "NVIDIA L4, simulation, no real robot"


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "uncommitted"


def meta() -> dict[str, Any]:
    """Provenance block attached to every results JSON."""
    return {
        "label": HARDWARE_LABEL,
        "gpu": gpu_name(),
        "host_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "git_rev": git_rev(),
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta(), **payload}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
