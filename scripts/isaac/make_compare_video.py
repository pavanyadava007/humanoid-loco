"""Side-by-side clip: CPU MuJoCo brax PPO + DR video (left) and Isaac Lab g1_flat video (right).

Two different policies, robot models and simulators, each in its own simulator; shown for a visual
impression only, not a comparison under matched conditions. Both clips are cut to the shorter length
and scaled to the same height. Writes media/compare_mujoco_vs_isaac.mp4 and results/compare_video.json.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hloco.common import MEDIA, RESULTS, ROOT, write_json  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_video import MAX_BYTES, probe  # noqa: E402

LEFT = MEDIA / "brax_dr_mujoco_cpu.mp4"
RIGHT = MEDIA / "isaac_g1_flat.mp4"
OUT = MEDIA / "compare_mujoco_vs_isaac.mp4"
HEIGHT = 360


def main() -> None:
    lp, rp = probe(LEFT), probe(RIGHT)
    dur = min(lp["duration_s"], rp["duration_s"])
    filt = (f"[0:v]fps=25,scale=-2:{HEIGHT},setsar=1,trim=duration={dur:.3f},setpts=PTS-STARTPTS[l];"
            f"[1:v]fps=25,scale=-2:{HEIGHT},setsar=1,trim=duration={dur:.3f},setpts=PTS-STARTPTS[r];"
            "[l][r]hstack=inputs=2[v]")
    for crf in (26, 30, 34):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(LEFT), "-i", str(RIGHT), "-filter_complex", filt,
                        "-map", "[v]", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-an", str(OUT)], check=True)
        if OUT.stat().st_size <= MAX_BYTES:
            break
    write_json(RESULTS / "compare_video.json", {
        "file": str(OUT.relative_to(ROOT)),
        "left": {"file": str(LEFT.relative_to(ROOT)), "what": "brax PPO + DR policy (Menagerie G1), plain CPU MuJoCo, sim-to-sim"},
        "right": {"file": str(RIGHT.relative_to(ROOT)), "what": "Isaac Lab RSL-RL PPO policy (Isaac Lab G1), Isaac Sim, training simulator"},
        "duration_s_each": dur,
        "height_px": HEIGHT,
        "crf": crf,
        "bytes": OUT.stat().st_size,
        **{f"out_{k}": v for k, v in probe(OUT).items()},
        "note": "different policies, robot models, commands and simulators; visual impression only",
    })
    print("wrote", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
