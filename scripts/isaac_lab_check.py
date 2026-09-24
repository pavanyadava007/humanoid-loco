"""Isaac Lab / Isaac Sim feasibility check on this host, without downloading Isaac Sim.

Reads the NVIDIA pip index listings (HTML only) to find which platform tags the isaacsim
wheels are built for, asks the server for wheel sizes with HEAD requests (and reads the
pinned requirements of the small isaacsim meta wheels), and compares with the host glibc,
Python, free disk and the 2 GB download cap. Writes results/isaac_lab_check.json.
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


def wheel_url(pkg: str, version: str) -> tuple[str, str] | None:
    """Linux x86_64 (or pure python) wheel of pkg==version on the NVIDIA index."""
    stem = pkg.replace("-", "_")
    for name, url in listing(pkg):
        if name.startswith(f"{stem}-{version}-") and ("manylinux" in name and "x86_64" in name or "none-any" in name):
            if "cp310" in name or "py3" in name:
                return name, url
    return None


def requires(url: str, extras: tuple[str, ...], max_bytes: int = 5_000_000) -> tuple[list[tuple[str, str]], int]:
    """Pinned requirements (name, version) of a small metadata-only wheel; returns (deps, bytes downloaded)."""
    import io
    import zipfile

    size = head_size(url)
    if size is None or size > max_bytes:
        return [], 0
    blob = urllib.request.urlopen(url, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    deps = []
    for line in zf.read(meta).decode().splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        spec = line.split(":", 1)[1].strip()
        req, _, marker = spec.partition(";")
        if marker and not any(f'extra == "{e}"' in marker for e in extras):
            continue
        if "==" in req:
            name, ver = req.split("==")
            deps.append((name.strip(), ver.strip()))
    return deps, len(blob)


def install_size(version: str = "4.5.0.0", extras: tuple[str, ...] = ("all", "extscache")) -> dict:
    """Sum of wheel sizes (HTTP HEAD) for `pip install isaacsim[all,extscache]==version`, NVIDIA-index packages only."""
    seen, sizes, missing, meta_bytes = set(), {}, [], 0
    queue = [("isaacsim", version)]
    while queue:
        pkg, ver = queue.pop()
        if pkg in seen:
            continue
        seen.add(pkg)
        try:
            w = wheel_url(pkg, ver)
        except Exception:  # noqa: BLE001  (not on this index)
            w = None
        if w is None:
            missing.append(f"{pkg}=={ver}")
            continue
        sizes[w[0]] = head_size(w[1])
        if pkg.startswith("isaacsim"):
            deps, nb = requires(w[1], extras if pkg == "isaacsim" else ())
            meta_bytes += nb
            queue += deps
    total = sum(v for v in sizes.values() if v)
    return {"isaacsim_version": version, "extras": list(extras), "wheels": sizes,
            "total_wheel_bytes": total, "not_found_on_index": missing,
            "metadata_wheel_bytes_downloaded": meta_bytes}


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

    try:
        inst45 = install_size()
    except Exception as e:  # noqa: BLE001
        inst45 = {"error": f"{type(e).__name__}: {e}"}
    try:
        inst45_noext = install_size(extras=("all",))
    except Exception as e:  # noqa: BLE001
        inst45_noext = {"error": f"{type(e).__name__}: {e}"}
    cap = 2 * 1024**3
    blockers = []
    if "total_wheel_bytes" in inst45 and inst45["total_wheel_bytes"] > cap:
        blockers.append(f"pip install isaacsim[all,extscache]==4.5.0.0 needs {inst45['total_wheel_bytes'] / 1e9:.1f} GB of wheels "
                        "from the NVIDIA index (sum of HEAD sizes), above the 2 GB download cap set for this check")
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
        "The variant without the extscache extras (size recorded separately) was not tried; the extension caches it "
        "leaves out would have to come from somewhere else at run time, so it is not a way around the cap that was verified.",
        "Install size counts only wheels hosted on the NVIDIA index (isaacsim-*, omniverse-kit); PyPI dependencies, "
        "Isaac Lab itself and runtime shader/asset caches are not included, so the real footprint is larger.",
    ]
    verdict = "not run: blocked under this project's constraints" if blockers else "no blocker found by this check"
    write_json(RESULTS / "isaac_lab_check.json", {
        "host_os": pretty,
        "host_glibc": libc[1],
        "host_python": platform.python_version(),
        "gpu_and_driver": drv,
        "free_disk_gb": round(free_gb, 1),
        "index": INDEX,
        "packages": pkgs,
        "isaacsim_4_5_install": inst45,
        "isaacsim_4_5_install_without_extscache": inst45_noext,
        "download_cap_bytes": cap,
        "wheel_bytes_downloaded": inst45.get("metadata_wheel_bytes_downloaded", 0),
        "blockers": blockers,
        "notes": notes,
        "verdict": verdict,
        "method": "HTML index listing + HTTP HEAD for sizes; only the small metadata wheels (isaacsim, isaacsim-* "
                  "meta packages under 5 MB) were downloaded to read their pinned requirements; nothing was installed",
    })
    print(verdict)
    for b in blockers:
        print(" -", b)


if __name__ == "__main__":
    main()
