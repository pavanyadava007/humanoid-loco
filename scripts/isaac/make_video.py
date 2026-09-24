"""Transcode the raw Isaac Lab play video to media/isaac_<name>.mp4 (H.264, yuv420p, faststart, <= 5 MB).

Reads the raw file path from results/isaac_play_<name>.json, writes results/isaac_video_<name>.json.
Stdlib + ffmpeg/ffprobe; run with either venv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hloco.common import MEDIA, RESULTS, ROOT, read_json, write_json  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024
WARMUP_TOL = 0.15  # a leading frame is warm-up if its mean luma is off the video median by more than 15%


def frame_luma(path: Path) -> list[float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-f", "lavfi", "-i", f"movie={path},signalstats",
                          "-show_entries", "frame_tags=lavfi.signalstats.YAVG", "-of", "csv=p=0"],
                         capture_output=True, text=True, check=True).stdout
    return [float(x.strip(",")) for x in out.split() if x.strip(",")]


def warmup_frames(luma: list[float], tol: float = WARMUP_TOL) -> int:
    """Number of leading frames whose mean luma is far from the median (renderer warm-up: black or flash)."""
    if not luma:
        return 0
    med = sorted(luma)[len(luma) // 2]
    for i, y in enumerate(luma):
        if abs(y - med) <= tol * med:
            return i
    return 0


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    return {"codec": st["codec_name"], "pix_fmt": st["pix_fmt"], "width": st["width"], "height": st["height"],
            "fps": st["r_frame_rate"], "frames": int(st.get("nb_frames", 0)), "duration_s": float(d["format"]["duration"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="g1_flat")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    play = read_json(RESULTS / f"isaac_play_{args.name}.json")
    raw = ROOT / play["video"]["raw_file"]
    out = MEDIA / f"isaac_{args.name}.mp4"
    MEDIA.mkdir(exist_ok=True)
    raw_info = probe(raw)
    num, den = (int(x) for x in raw_info["fps"].split("/"))
    luma = frame_luma(raw)
    n_trim = warmup_frames(luma)
    start_s = n_trim * den / num
    for crf in (24, 28, 32, 36):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start_s:.4f}", "-i", str(raw), "-vf", f"fps={args.fps}",
                        "-c:v", "libx264",
                        "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                        str(out)], check=True)
        if out.stat().st_size <= MAX_BYTES:
            break
    info = probe(out)
    write_json(RESULTS / f"isaac_video_{args.name}.json", {
        "file": str(out.relative_to(ROOT)),
        "source": "Isaac Lab RecordVideo (rgb_array render of the viewport camera), transcoded with ffmpeg",
        "raw_file": play["video"]["raw_file"],
        "raw_probe": raw_info,
        "trimmed_leading_frames": n_trim,
        "trimmed_start_s": start_s,
        "trim_rule": f"leading raw frames whose mean luma differs from the video median by more than {WARMUP_TOL:.0%} "
                     "(renderer warm-up)",
        "leading_raw_frame_luma": luma[: n_trim + 3],
        "crf": crf,
        "bytes": out.stat().st_size,
        **info,
    })
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
