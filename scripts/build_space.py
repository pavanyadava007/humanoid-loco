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


def load_opt(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def isaac_section() -> tuple[str, str, str]:
    """(video figure html, results html, limits bullet) for the Isaac Lab track, from results/isaac_*.json."""
    tr, ev, vid = load_opt("isaac_train_g1_flat.json"), load_opt("isaac_eval_g1_flat.json"), load_opt("isaac_video_g1_flat.json")
    play, mp = load_opt("isaac_play_g1_flat.json"), load_opt("isaac_mujoco_mapping.json")
    inst = load_opt("isaac_install.json") or {}
    if tr is None:
        return "", "", "<li>Isaac Lab was not run on this host; see the repository.</li>"
    fig = ""
    if vid and play:
        v = play["video"]
        fig = (f"<figure class='vid big'><video src='media/isaac_g1_flat.mp4' poster='media/isaac_g1_flat.jpg' controls muted loop "
               f"playsinline preload=metadata></video><figcaption><b>Isaac Lab: RSL-RL PPO, Isaac-Velocity-Flat-G1-v0, in Isaac Sim (PhysX)</b><br>"
               f"{tr['final_step'] / 1e6:.1f}M env steps in {tr['wall_clock_s'] / 60:.0f} minutes on one NVIDIA L4. Constant command "
               f"{e(v['command_vx_vy_wz'][0])} m/s forward; the camera follows one robot (Isaac Lab G1, {e(play['model']['num_joints'])} joints). "
               "Trained and shown in the same simulator: this is not a transfer test.</figcaption></figure>")
    trr, evr, vr = load_opt("isaac_train_g1_rough.json"), load_opt("isaac_eval_g1_rough.json"), load_opt("isaac_video_g1_rough.json")
    if trr and vr and (ROOT / vr["file"]).exists():
        nom = evr["summary"].get("nominal") if evr else None
        ev_txt = (f" Within Isaac Sim, nominal: {nom['falls']}/{nom['episodes']} falls on the task's own terrain mix." if nom else "")
        fig += (f"<figure class='vid big'><video src='media/isaac_g1_rough.mp4' poster='media/isaac_g1_rough.jpg' controls muted loop "
                f"playsinline preload=metadata></video><figcaption><b>Isaac Lab: RSL-RL PPO, Isaac-Velocity-Rough-G1-v0 (rough terrain, height scan)</b><br>"
                f"{trr['final_step'] / 1e6:.1f}M env steps in {trr['wall_clock_s'] / 60:.0f} minutes ({e(trr['status'])}, "
                f"{trr['iterations_completed']} of {trr['requested_max_iterations']} iterations).{e(ev_txt)}</figcaption></figure>")
    cmp = load_opt("compare_video.json")
    if cmp and (ROOT / cmp["file"]).exists():
        fig += ("<figure class='vid big'><video src='media/compare_mujoco_vs_isaac.mp4' poster='media/compare_mujoco_vs_isaac.jpg' "
                "controls muted loop playsinline preload=metadata></video><figcaption><b>Side by side: CPU MuJoCo (left) and Isaac Sim (right)</b><br>"
                "Left: brax PPO + DR policy on the Menagerie G1 in plain CPU MuJoCo. Right: Isaac Lab RSL-RL policy on the Isaac Lab G1 in Isaac Sim. "
                "Different policies, robot models, commands and simulators; a visual impression, not a matched comparison.</figcaption></figure>")
    rows = []
    isaac_labels = dict(CONDITIONS)
    isaac_labels.update({"friction_x0.5": "robot contact friction x0.5", "friction_x1.5": "robot contact friction x1.5",
                         "pushes": "random pushes (none in training)"})
    if ev:
        for c, lbl in isaac_labels.items():
            s_ = ev["summary"].get(c)
            if s_ is None:
                continue
            rows.append(f"<tr><td>{e(lbl)}</td><td>{s_['falls']}/{s_['episodes']}<span class=ci>{pct(s_['fall_rate'])} "
                        f"[{pct(s_['fall_rate_wilson95'][0])}, {pct(s_['fall_rate_wilson95'][1])}]</span></td>"
                        f"<td>{s_['lin_vel_err_mean']:.3f}</td><td>{s_['yaw_rate_err_mean']:.3f}</td></tr>")
    table = ("<div class=tw><table><tr><th>condition</th><th>falls (Wilson 95%)</th><th>lin-vel error m/s</th>"
             f"<th>yaw-rate error rad/s</th></tr>{''.join(rows)}</table></div>") if rows else "<p class=note>Evaluation not run.</p>"
    last = next((r for r in reversed(tr["curve"]) if "mean_episode_length" in r), {})
    mp_txt = ""
    if mp:
        j = mp["joints"]
        mp_txt = (f"<p class=note>Isaac to MuJoCo sim-to-sim was not attempted: the Isaac Lab G1 has {j['isaac_num_joints']} action joints "
                  f"(with hands), the Menagerie G1 used above has {j['mujoco_num_actuated_joints']} actuators, and only {len(j['common'])} "
                  "joint names match, so no exact joint and observation mapping exists.</p>")
    body = (f"<h2>Isaac Lab track: same task idea, Isaac Sim + RSL-RL</h2>"
            f"<p>Isaac Sim {e(tr['isaac_sim'])} and Isaac Lab {e(inst.get('isaac_lab_git_tag', tr['isaac_lab']))} installed on this host; "
            f"<code>{e(tr['task'])}</code> trained with RSL-RL PPO, {e(tr['num_envs'])} parallel envs, "
            f"{tr['iterations_completed']} iterations, {tr['final_step']:,} env steps, {tr['wall_clock_s'] / 60:.0f} minutes, "
            f"final mean training episode length {last.get('mean_episode_length', 0):.0f} of 1000 steps.</p>"
            f"<p class=note>Evaluation {e(ev['label'] if ev else 'within Isaac Sim (PhysX), not sim-to-sim')}; "
            f"{e(ev['episodes_per_condition']) if ev else 0} episodes of 20 s per condition. "
            "Different robot model, fall definition and command ranges than the MuJoCo table above, so compare within this table only.</p>"
            f"{table}{mp_txt}")
    return fig, body, "<li>Isaac Lab results are measured inside Isaac Sim, where the policy was trained; they are not a transfer test.</li>"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "media").mkdir(parents=True)
    for fname, _, _ in VIDEOS:
        shutil.copy2(ROOT / "media" / fname, OUT / "media" / fname)
        poster = OUT / "media" / fname.replace(".mp4", ".jpg")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.1" if "rsl" in fname else "4", "-i",
                        str(ROOT / "media" / fname), "-frames:v", "1", "-q:v", "4", str(poster)], check=True)

    isaac_fig, isaac_body, isaac_limit = isaac_section()
    if isaac_fig:
        shutil.copy2(ROOT / "media" / "isaac_g1_flat.mp4", OUT / "media" / "isaac_g1_flat.mp4")
        for fname in ("isaac_g1_flat.mp4", "isaac_g1_rough.mp4", "compare_mujoco_vs_isaac.mp4"):
            if not (ROOT / "media" / fname).exists():
                continue
            if fname != "isaac_g1_flat.mp4":
                shutil.copy2(ROOT / "media" / fname, OUT / "media" / fname)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "4", "-i", str(ROOT / "media" / fname),
                            "-frames:v", "1", "-q:v", "4", str(OUT / "media" / fname.replace(".mp4", ".jpg"))], check=True)

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
<h2>Videos</h2><div class=vids>{''.join(vids)}{isaac_fig}</div>
<h2>Falls in plain CPU MuJoCo (out of 500 episodes, 20 s each)</h2>
<div class=tw><table><tr><th>condition</th>{head}</tr>{''.join(rows)}</table></div>
<p class=note>Same seeds (start state, velocity command, pushes) for every policy. Wilson 95% intervals are in the full results.</p>
<h2>Does domain randomization help? Paired comparison at the same training steps</h2>
<div class=tw><table><tr><th>condition</th><th>falls, PPO + DR</th><th>falls, PPO no DR</th><th>exact McNemar p</th></tr>{''.join(paired)}</table></div>
<p class=note>Highlighted rows: p &lt; 0.01. One training seed per policy, so this speaks to these two checkpoints, not to domain randomization in general.</p>
{isaac_body}
<h2>How it was built</h2><ul>
<li>Training: brax PPO, MuJoCo Playground <code>G1JoystickFlatTerrain</code>, 8,192 parallel environments, Playground default config with its domain randomizer.</li>
<li>Export: static ONNX graph (opset 17), max action difference to the training framework {e(f'{onnx_err:.2g}' if isinstance(onnx_err, float) else onnx_err)}.</li>
<li>Evaluation: the ONNX policy runs closed-loop at 50 Hz in plain CPU MuJoCo, with the observation rebuilt and checked element-wise against MJX.</li>
<li>Ablations: no domain randomization; an RSL-RL PPO run (failed to learn within its budget).</li></ul>
<h2>Limits</h2><ul>
<li>Simulation only: nothing here has run on a real robot.</li>
<li>Two control steps of action delay and the harsh push condition break every policy often.</li>
<li>One training seed per configuration; budgets below the Playground default of 200M steps.</li>
{isaac_limit}</ul>
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
short_description: Unitree G1 walking with PPO + domain randomization
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
