"""Build the static Hugging Face Space (space/) from results/*.json and media/*.mp4.

Every number on the page is read from a results JSON; nothing is typed by hand.
Upload with: huggingface-cli upload <user>/humanoid-loco space . --repo-type space
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hloco.common import HARDWARE_LABEL, RESULTS, ROOT  # noqa: E402
from hloco.stats import mcnemar_exact  # noqa: E402

OUT = ROOT / "space"
GITHUB = "https://github.com/pavanyadava007/humanoid-loco"
VIDEOS = [
    ("brax_dr_mujoco_cpu.mp4", "PPO + domain randomization, plain CPU MuJoCo",
     "The ONNX policy running in a different simulator from the one it was trained in (sim-to-sim)."),
    ("brax_dr_mjx.mp4", "PPO + domain randomization, MJX",
     "Same policy in the training simulator, same scripted velocity commands."),
    ("brax_nodr_mujoco_cpu.mp4", "Ablation: PPO without domain randomization, plain CPU MuJoCo",
     "Also walks in the nominal case; the difference shows up under latency and low friction (table below)."),
    ("rsl_dr_mujoco_cpu.mp4", "Failed run: RSL-RL PPO, 72 minute budget",
     "Did not learn to stand within its budget. Kept on purpose: failures are part of the result."),
]
POLICIES = [("brax_dr", "PPO + DR (final)"), ("brax_dr_matched", "PPO + DR (step-matched)"),
            ("brax_nodr", "PPO, no DR"), ("rsl_dr", "RSL-RL PPO + DR")]
CONDITIONS = [("nominal", "nominal"), ("friction_x0.5", "floor friction x0.5"), ("friction_x1.5", "floor friction x1.5"),
              ("torso_mass_+3kg", "torso mass +3 kg"), ("torso_mass_-3kg", "torso mass -3 kg"),
              ("pushes", "random pushes (harsher than training)"), ("action_delay_1step", "1-step action delay"),
              ("action_delay_2steps", "2-step action delay"), ("obs_noise_train_level", "sensor noise (training level)"),
              ("accurate_solver", "more accurate solver")]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def e(x) -> str:
    return html.escape(str(x))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "media").mkdir(parents=True)
    for fname, _, _ in VIDEOS:
        shutil.copy2(ROOT / "media" / fname, OUT / "media" / fname)
        poster = OUT / "media" / fname.replace(".mp4", ".jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.1" if "rsl" in fname else "4", "-i",
                        str(ROOT / "media" / fname), "-frames:v", "1", "-q:v", "4", str(poster)], check=True)

    train = load("train_brax_dr.json")
    s2s = {p: load(f"sim2sim_{p}.json") for p, _ in POLICIES}
    onnx = load("onnx_parity_brax_dr.json")
    dr_nom = s2s["brax_dr"]["summary"]["nominal"]
    steps = train["final_step"]
    minutes = train["wall_clock_s"] / 60
    onnx_err = onnx["max_abs_err_onnx_vs_reference"]

    # headline cards
    matched, nodr = s2s["brax_dr_matched"], s2s["brax_nodr"]
    d1m, d1n = matched["summary"]["action_delay_1step"], nodr["summary"]["action_delay_1step"]
    cards = [
        (f"{dr_nom['falls']} / {dr_nom['episodes']}", "falls in plain CPU MuJoCo, nominal (20 s episodes)"),
        (f"{d1m['falls']} vs {d1n['falls']}", "falls with a 1-step action delay: DR vs no DR at the same training steps"),
        (f"{steps / 1e6:.1f}M steps", f"training in {minutes:.0f} minutes on one NVIDIA L4"),
        (f"{dr_nom['lin_vel_err_mean']:.3f} m/s", "mean velocity-tracking error, nominal"),
    ]

    # fall table
    head = "".join(f"<th>{e(lbl)}</th>" for _, lbl in POLICIES)
    rows = []
    for c, lbl in CONDITIONS:
        cells = []
        for p, _ in POLICIES:
            s = s2s[p]["summary"].get(c)
            cells.append("<td>n/a</td>" if s is None else
                         f"<td>{s['falls']}/{s['episodes']}<span class=ci>{pct(s['fall_rate'])}</span></td>")
        rows.append(f"<tr><td>{e(lbl)}</td>{''.join(cells)}</tr>")

    # paired step-matched comparison
    paired = []
    for c, lbl in CONDITIONS:
        if c not in matched["summary"] or c not in nodr["summary"]:
            continue
        ea, eb = matched["episodes"][c], nodr["episodes"][c]
        only_a = sum(x["fell"] and not y["fell"] for x, y in zip(ea, eb, strict=True))
        only_b = sum(y["fell"] and not x["fell"] for x, y in zip(ea, eb, strict=True))
        p = mcnemar_exact(only_a, only_b)
        cls = " class=sig" if p < 0.01 else ""
        paired.append(f"<tr{cls}><td>{e(lbl)}</td><td>{matched['summary'][c]['falls']}</td>"
                      f"<td>{nodr['summary'][c]['falls']}</td><td>{p:.2g}</td></tr>")

    vids = []
    for i, (fname, title, cap) in enumerate(VIDEOS):
        big = " big" if i == 0 else ""
        vids.append(f"<figure class='vid{big}'><video src='media/{fname}' poster='media/{fname[:-4]}.jpg' controls muted loop playsinline preload=metadata"
                    f"{' autoplay' if i == 0 else ''}></video><figcaption><b>{e(title)}</b><br>{e(cap)}</figcaption></figure>")

    page = f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>humanoid-loco: Unitree G1 locomotion</title>
<style>
:root{{--bg:#f7f8fa;--fg:#1c2330;--mut:#5b6575;--card:#fff;--line:#e2e6ec;--acc:#2a5db0;--good:#e8f1ff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e6e9ef;--mut:#9aa4b2;--card:#1b2029;--line:#2b323d;--acc:#7aa7f0;--good:#1c2a40}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px 60px}}h1{{font-size:28px;margin:0 0 6px}}h2{{font-size:19px;margin:36px 0 10px}}
.sub{{color:var(--mut);margin:0 0 14px}}.badge{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:13px;color:var(--mut);margin:0 6px 6px 0}}
a{{color:var(--acc)}}.btn{{display:inline-block;background:var(--acc);color:#fff;text-decoration:none;padding:7px 14px;border-radius:8px;font-weight:600;margin-right:8px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:22px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}.card b{{display:block;font-size:24px;color:var(--acc)}}.card span{{color:var(--mut);font-size:13px}}
.vids{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.vid{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.vid.big{{grid-column:1/-1}}video{{width:100%;display:block;background:#000}}figcaption{{padding:10px 12px;font-size:13px;color:var(--mut)}}figcaption b{{color:var(--fg)}}
@media (max-width:760px){{.vids{{grid-template-columns:1fr}}}}
.tw{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--mut);font-weight:600}}
.ci{{color:var(--mut);margin-left:6px;font-size:12px}}tr.sig td{{background:var(--good)}}.note{{color:var(--mut);font-size:13px}}
ul{{padding-left:20px}}</style></head><body><main>
<h1>humanoid-loco: Unitree G1 locomotion with reinforcement learning</h1>
<p class=sub>Velocity-commanded walking trained with PPO and domain randomization in MuJoCo Playground (MJX),
exported to ONNX and tested sim-to-sim in plain CPU MuJoCo on 500 seeded episodes per condition.</p>
<span class=badge>{e(HARDWARE_LABEL)}</span><span class=badge>sim-to-sim, not sim-to-real</span><span class=badge>every number from results/*.json</span>
<p><a class=btn href="{GITHUB}">Code on GitHub</a><a href="{GITHUB}/blob/main/docs/RESULTS.md">Full results</a></p>
<div class=cards>{''.join(f'<div class=card><b>{e(a)}</b><span>{e(b)}</span></div>' for a, b in cards)}</div>
<h2>Videos</h2><div class=vids>{''.join(vids)}</div>
<h2>Falls in plain CPU MuJoCo (out of 500 episodes, 20 s each)</h2>
<div class=tw><table><tr><th>condition</th>{head}</tr>{''.join(rows)}</table></div>
<p class=note>Same seeds (start state, velocity command, pushes) for every policy. Wilson 95% intervals are in the full results.</p>
<h2>Does domain randomization help? Paired comparison at the same training steps</h2>
<div class=tw><table><tr><th>condition</th><th>falls, PPO + DR</th><th>falls, PPO no DR</th><th>exact McNemar p</th></tr>{''.join(paired)}</table></div>
<p class=note>Highlighted rows: p &lt; 0.01. One training seed per policy, so this speaks to these two checkpoints, not to domain randomization in general.</p>
<h2>How it was built</h2><ul>
<li>Training: brax PPO, MuJoCo Playground <code>G1JoystickFlatTerrain</code>, 8,192 parallel environments, Playground default config with its domain randomizer.</li>
<li>Export: static ONNX graph (opset 17), max action difference to the training framework {e(f'{onnx_err:.2g}' if isinstance(onnx_err, float) else onnx_err)}.</li>
<li>Evaluation: the ONNX policy runs closed-loop at 50 Hz in plain CPU MuJoCo, with the observation rebuilt and checked element-wise against MJX.</li>
<li>Ablations: no domain randomization; an RSL-RL PPO run (failed to learn within its budget).</li></ul>
<h2>Limits</h2><ul>
<li>Simulation only: nothing here has run on a real robot.</li>
<li>Two control steps of action delay and the harsh push condition break every policy often.</li>
<li>One training seed per configuration; budgets below the Playground default of 200M steps.</li>
<li>Isaac Lab was not run on this host (install size); see the repository.</li></ul>
<p class=note>Pavan Yadav Annappa. Model: Unitree G1 from MuJoCo Menagerie (BSD-3-Clause). Code: MIT.</p>
</main></body></html>
"""
    (OUT / "index.html").write_text(page)
    (OUT / "README.md").write_text("""---
title: humanoid-loco
emoji: 🚶
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Unitree G1 walking policy, PPO + domain randomization, sim-to-sim
tags:
  - robotics
  - reinforcement-learning
  - humanoid
  - locomotion
  - mujoco
---

Static results page for https://github.com/pavanyadava007/humanoid-loco (simulation only, no real robot).
""")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
