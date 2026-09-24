"""Isaac Lab / Isaac Sim feasibility check on this host, without downloading Isaac Sim.

Reads the NVIDIA pip index listings (HTML only) to find which platform tags the isaacsim
wheels are built for, asks the server for wheel sizes with HEAD requests, and compares
with the host glibc, Python and free disk. Writes results/isaac_lab_check.json.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hloco.common import RESULTS, write_json  # noqa: E402

INDEX = "https://pypi.nvidia.com"
PACKAGES = ["isaacsim", "isaacsim-kernel"]


def listing(pkg: str) -> list[tuple[str, str]]:
    html = urllib.request.urlopen(f"{INDEX}/{pkg}/", timeout=60).read().decode()
    out = []
    for href, name in re.findall(r'href="([^"]+)"[^>]*>([^<]+\.whl)<', html):
        out.append((name.strip(), urllib.request.urljoin(f"{INDEX}/{pkg}/", href.split("#")[0])))
    return out


def head_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            ctype = r.headers.get("Content-Type", "")
            if r.status != 200 or "html" in ctype or not n:
                return None  # proxy/error page, not the wheel
            return int(n)
    except Exception:  # noqa: BLE001
        return None


def glibc_tag(name: str) -> tuple[int, int] | None:
    m = re.search(r"manylinux_(\d+)_(\d+)_x86_64", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main() -> None:
    libc = platform.libc_ver()
    host_glibc = tuple(int(x) for x in libc[1].split(".")[:2]) if libc[1] else None
    free_gb = shutil.disk_usage(Path.home()).free / 1e9
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        drv = "unknown"
    os_release = Path("/etc/os-release").read_text().splitlines()
    pretty = next((ln.split("=", 1)[1].strip('"') for ln in os_release if ln.startswith("PRETTY_NAME=")), "unknown")

    pkgs = {}
    for pkg in PACKAGES:
        try:
            wheels = [(n, u) for n, u in listing(pkg) if "x86_64" in n and "manylinux" in n]
        except Exception as e:  # noqa: BLE001
            pkgs[pkg] = {"error": f"{type(e).__name__}: {e}"}
            continue
        tags = sorted({glibc_tag(n) for n, _ in wheels if glibc_tag(n)})
        min_tag = tags[0] if tags else None
        latest = wheels[-1] if wheels else None
        inst = [(n, u) for n, u in wheels if host_glibc and glibc_tag(n) and glibc_tag(n) <= host_glibc]
        installable = [n for n, _ in inst]
        pkgs[pkg] = {
            "linux_x86_64_wheels_listed": len(wheels),
            "manylinux_glibc_tags": [f"{a}.{b}" for a, b in tags],
            "lowest_glibc_required": f"{min_tag[0]}.{min_tag[1]}" if min_tag else None,
            "wheels_installable_on_host_glibc": installable,
            "latest_wheel": latest[0] if latest else None,
            "latest_wheel_bytes_HEAD": head_size(latest[1]) if latest else None,
            "newest_host_installable_wheel": inst[-1][0] if inst else None,
            "newest_host_installable_wheel_bytes_HEAD": head_size(inst[-1][1]) if inst else None,
            "python_tags": sorted({n.split("-")[2] for n, _ in wheels}),
        }

    blockers = []
    for pkg, info in pkgs.items():
        if "error" in info:
            blockers.append(f"{pkg}: index not reachable ({info['error']})")
        elif not info["wheels_installable_on_host_glibc"]:
            blockers.append(f"{pkg}: every Linux x86_64 wheel needs glibc >= {info['lowest_glibc_required']}, "
                            f"host has {libc[1]}")
    if free_gb < 30:
        blockers.append(f"free disk {free_gb:.1f} GB, below the 30 GB threshold this check assumes for Isaac Sim + "
                        "Isaac Lab + shader/extension caches (assumption, not measured); the brief also caps "
                        "Isaac downloads at 2 GB")
    notes = [
        "Isaac Sim 4.x wheels are tagged manylinux_2_34 (cp310), so glibc 2.34 alone does not block pip install of 4.x; "
        "5.x/6.x wheels need glibc 2.35. The Isaac Sim documentation lists Ubuntu and Windows as supported OSes; Amazon Linux 2023 is not listed: "
        "runtime compatibility on this OS was not tested.",
        "Only the isaacsim and isaacsim-kernel wheel sizes were measured (HEAD). A full Isaac Sim pip install pulls many "
        "more isaacsim-* packages plus extension caches; their total size was not measured here.",
    ]
    verdict = "not feasible on this host" if blockers else "no blocker found by this check"
    write_json(RESULTS / "isaac_lab_check.json", {
        "host_os": pretty,
        "host_glibc": libc[1],
        "host_python": platform.python_version(),
        "gpu_and_driver": drv,
        "free_disk_gb": round(free_gb, 1),
        "index": INDEX,
        "packages": pkgs,
        "wheel_bytes_downloaded": 0,
        "blockers": blockers,
        "notes": notes,
        "verdict": verdict,
        "method": "HTML index listing + HTTP HEAD only; no wheel was downloaded or installed",
    })
    print(verdict)
    for b in blockers:
        print(" -", b)


if __name__ == "__main__":
    main()
