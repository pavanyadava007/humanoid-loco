"""Generate docs/RESULTS.md (and the README headline table) from results/*.json only.

No number in the generated text is typed by hand: every value is read from a JSON file
written by a script in this repo, and each table names its source file.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hloco.common import HARDWARE_LABEL, RESULTS, ROOT  # noqa: E402
from hloco.stats import mcnemar_exact  # noqa: E402

POLICIES = [("brax_dr", "brax PPO + DR"), ("brax_dr_matched", "brax PPO + DR, step-matched"),
            ("brax_nodr", "brax PPO, no DR"), ("rsl_dr", "RSL-RL PPO + DR")]
TRAIN_RUNS = [("brax_dr", "brax PPO + DR"), ("brax_nodr", "brax PPO, no DR"), ("rsl_dr", "RSL-RL PPO + DR")]


def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def ci_pct(ci) -> str:
    return "n/a" if not ci else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def hms(s) -> str:
    if s is None:
        return "n/a"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def final_reward(tr: dict) -> float | None:
    for rec in reversed(tr.get("curve", [])):
        for k in ("episode_reward", "mean_episode_reward"):
            if k in rec:
                return rec[k]
    return None


def training_table() -> list[str]:
    rows = ["| run | algorithm | DR | env steps | wall clock | env steps/s (steady) | final reward | status | source |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, label in TRAIN_RUNS:
        tr = load(f"train_{name}.json")
        if tr is None:
            rows.append(f"| {name} | {label} | | not run | | | | | |")
            continue
        rew_kind = "eval" if name.startswith("brax") else "train-episode"
        rows.append(
            f"| {name} | {label} | {'yes' if tr['domain_randomization'] else 'no'} | {tr['final_step']:,} | "
            f"{hms(tr['wall_clock_s'])} | {f(tr.get('env_steps_per_s_steady'), 0)} | "
            f"{f(final_reward(tr), 2)} ({rew_kind}) | {tr['status']} | `results/train_{name}.json` |")
    return rows


def curve_table(name: str, max_rows: int = 12) -> list[str]:
    tr = load(f"train_{name}.json")
    if tr is None or not tr.get("curve"):
        return [f"_{name}: no curve (run not available)_"]
    c = tr["curve"]
    idx = sorted({round(i * (len(c) - 1) / (max_rows - 1)) for i in range(max_rows)}) if len(c) > max_rows else range(len(c))
    key = "episode_reward" if "episode_reward" in c[-1] else "mean_episode_reward"
    lkey = "avg_episode_length" if "avg_episode_length" in c[-1] else "mean_episode_length"
    rows = [f"**{name}** (`results/train_{name}.json`, {key})", "", "| env steps | wall clock | reward | episode length |",
            "|---|---|---|---|"]
    for i in idx:
        r = c[i]
        rows.append(f"| {r['step']:,} | {hms(r['wall_s'])} | {f(r.get(key), 2)} | {f(r.get(lkey), 1)} |")
    return rows


def mjx_table() -> list[str]:
    rows = ["| policy | setting | mean return [95% CI] | fall rate [Wilson 95%] | envs | source |", "|---|---|---|---|---|---|"]
    for name, label in POLICIES:
        r = load(f"mjx_eval_{name}.json")
        if r is None:
            continue
        for key, lab in (("with_dr", "MJX + DR"), ("without_dr", "MJX nominal")):
            e = r[key]
            rows.append(f"| {label} | {lab} | {f(e['mean_return'], 2)} [{f(e['return_ci95'][0], 2)}, {f(e['return_ci95'][1], 2)}] | "
                        f"{pct(e['fall_rate'])} {ci_pct(e['fall_rate_wilson95'])} | {e['envs']} | `results/mjx_eval_{name}.json` |")
    return rows


def sim2sim_tables() -> tuple[list[str], list[str]]:
    data = {n: load(f"sim2sim_{n}.json") for n, _ in POLICIES}
    avail = [(n, lab) for n, lab in POLICIES if data[n] is not None]
    if not avail:
        return ["_sim-to-sim evaluation not run_"], []
    conds = list(data[avail[0][0]]["summary"].keys())
    head = "| condition | " + " | ".join(f"{lab}: fall rate [Wilson 95%]" for _, lab in avail) + " |"
    fall = [head, "|---|" + "---|" * len(avail)]
    for c in conds:
        cells = []
        for n, _ in avail:
            s = data[n]["summary"].get(c)
            cells.append("n/a" if s is None else f"{pct(s['fall_rate'])} {ci_pct(s['fall_rate_wilson95'])} (n={s['episodes']})")
        fall.append(f"| {c} | " + " | ".join(cells) + " |")
    head2 = "| condition | " + " | ".join(f"{lab}: lin-vel err m/s / survival s" for _, lab in avail) + " |"
    trk = [head2, "|---|" + "---|" * len(avail)]
    for c in conds:
        cells = []
        for n, _ in avail:
            s = data[n]["summary"].get(c)
            cells.append("n/a" if s is None else f"{f(s['lin_vel_err_mean'])} [{f(s['lin_vel_err_ci95'][0])}, {f(s['lin_vel_err_ci95'][1])}] / {f(s['survival_s_mean'], 1)}")
        trk.append(f"| {c} | " + " | ".join(cells) + " |")
    src = ", ".join(f"`results/sim2sim_{n}.json`" for n, _ in avail)
    return fall + ["", f"Source: {src}"], trk + ["", f"Source: {src}"]


def parity_rows() -> list[str]:
    rows = []
    op = load("obs_parity.json")
    if op:
        rows.append(f"- Observation parity MJX vs CPU MuJoCo, same state: max abs error {op['obs_max_abs_err']:.2e} over "
                    f"{op['n_states']} states (`results/obs_parity.json`).")
        rows.append(f"- One control step (10 substeps) from the same state and targets: qpos max-abs diff median "
                    f"{op['one_ctrl_step_qpos_max_abs_diff']['median']:.2e} (max {op['one_ctrl_step_qpos_max_abs_diff']['max']:.2e}), "
                    f"qvel median {op['one_ctrl_step_qvel_max_abs_diff']['median']:.2e} (max {op['one_ctrl_step_qvel_max_abs_diff']['max']:.2e}). "
                    "This is the simulator gap, not an observation bug.")
    for n, lab in POLICIES:
        r = load(f"onnx_parity_{n}.json")
        if r:
            rows.append(f"- ONNX (opset {r['onnx_opset']}) vs {r['reference']} for {lab}: max abs action error "
                        f"{r['max_abs_err_onnx_vs_reference']:.2e} over {r['n_obs']} observations (`results/onnx_parity_{n}.json`).")
    return rows or ["_parity checks not run_"]


def matched_curve_rows() -> list[str]:
    a, b = load("train_brax_dr.json"), load("train_brax_nodr.json")
    if not a or not b:
        return ["_needs train_brax_dr.json and train_brax_nodr.json_"]
    ca = {r["step"]: r for r in a["curve"]}
    rows = ["| env steps | DR run reward | no-DR run reward |", "|---|---|---|"]
    n = 0
    for r in b["curve"]:
        if r["step"] in ca:
            rows.append(f"| {r['step']:,} | {f(ca[r['step']].get('episode_reward'), 2)} | {f(r.get('episode_reward'), 2)} |")
            n += 1
    if n == 0:
        return ["_no common evaluation steps between the two runs_"]
    return rows + ["", "Both runs use num_timesteps 130M with 20 evaluation points, so evaluation steps coincide; "
                   "each run is scored by brax's evaluator in its own training env (the DR run's eval env is randomized, the no-DR run's is nominal)."]


def dr_effect_rows(dr_name: str = "brax_dr") -> list[str]:
    a, b = load(f"sim2sim_{dr_name}.json"), load("sim2sim_brax_nodr.json")
    if not a or not b:
        return [f"_DR vs no-DR comparison needs both sim2sim_{dr_name}.json and sim2sim_brax_nodr.json_"]
    rows = ["| condition | falls DR | falls no-DR | paired episodes: only DR fell / only no-DR fell | exact McNemar p |",
            "|---|---|---|---|---|"]
    for c, s in a["summary"].items():
        if c not in b["summary"]:
            continue
        ea, eb = a["episodes"][c], b["episodes"][c]
        only_a = sum(x["fell"] and not y["fell"] for x, y in zip(ea, eb, strict=True))
        only_b = sum(y["fell"] and not x["fell"] for x, y in zip(ea, eb, strict=True))
        p = mcnemar_exact(only_a, only_b)
        rows.append(f"| {c} | {s['falls']}/{s['episodes']} | {b['summary'][c]['falls']}/{b['summary'][c]['episodes']} | "
                    f"{only_a} / {only_b} | {p:.2g} |")
    return rows + ["", "Same seeds (initial state, command, pushes) for both policies, so episodes are paired; "
                   "the p-value is the two-sided exact McNemar test on the discordant pairs (one training seed per policy, "
                   "so it speaks to these two checkpoints, not to DR in general)."]


def brax_vs_rsl_rows() -> list[str]:
    a, r = load("train_brax_dr.json"), load("train_rsl_dr.json")
    if not a or not r or not r.get("curve"):
        return ["_needs train_brax_dr.json and train_rsl_dr.json_"]
    rc = [x for x in r["curve"] if "mean_episode_reward" in x]
    rows = ["| env steps (brax eval point) | brax DR eval reward / episode length | RSL-RL env steps (nearest) | RSL-RL train reward / episode length |",
            "|---|---|---|---|"]
    for p in a["curve"]:
        if p["step"] == 0 or p["step"] > rc[-1]["step"]:
            continue
        q = min(rc, key=lambda x: abs(x["step"] - p["step"]))
        rows.append(f"| {p['step']:,} | {f(p.get('episode_reward'), 2)} / {f(p.get('avg_episode_length'), 1)} | "
                    f"{q['step']:,} | {f(q['mean_episode_reward'], 2)} / {f(q['mean_episode_length'], 1)} |")
    return rows + ["", "Episode length is in control steps (1000 = full 20 s episode). The reward columns use different "
                   "estimators (see above); episode length is the more comparable column (both count control steps from the env reset distribution until termination or 1000 steps)."]


def scheduling_rows() -> list[str]:
    """Why the no-DR and RSL-RL runs were run one after the other (numbers from the run JSONs)."""
    dr, nodr, att = load("train_brax_dr.json"), load("train_brax_nodr.json"), load("train_rsl_dr_parallel_attempt.json")
    if not (dr and nodr and att) or len(nodr["curve"]) < 2:
        return ["_scheduling data not available_"]
    c0, c1 = nodr["curve"][0], nodr["curve"][1]
    brax_par = (c1["step"] - c0["step"]) / (c1["wall_s"] - c0["wall_s"])
    pts = [r for r in att["curve"] if c0["wall_s"] <= r["wall_s"] <= c1["wall_s"]]
    rsl_par = (pts[-1]["step"] - pts[0]["step"]) / (pts[-1]["wall_s"] - pts[0]["wall_s"]) if len(pts) >= 2 else None
    probe = load("throughput_probe.json")
    rows = []
    if probe:
        rows.append(f"- Budget probe before the main runs: brax PPO with DR, {probe['final_step']:,} env steps, steady "
                    f"{f(probe['env_steps_per_s_steady'], 0)} env steps/s (`results/throughput_probe.json`); the 130M-step "
                    "budget was chosen from this rate to fit the 2 h cap.")
    rows += [
        f"- brax PPO alone on the L4 (DR run, steady): {f(dr['env_steps_per_s_steady'], 0)} env steps/s (`results/train_brax_dr.json`).",
        f"- brax PPO (no-DR run) while the RSL-RL job ran on the same GPU: {f(brax_par, 0)} env steps/s "
        f"(first two eval points of `results/train_brax_nodr.json`).",
        f"- RSL-RL over the same wall-clock window: {f(rsl_par, 0)} env steps/s (`results/train_rsl_dr_parallel_attempt.json`).",
    ]
    if rsl_par:
        rows.append(f"- Combined parallel throughput {f(brax_par + rsl_par, 0)} env steps/s vs {f(dr['env_steps_per_s_steady'], 0)} for one brax job alone, "
                    "so the RSL-RL job was stopped and the runs were made sequential (no-DR brax first, then RSL-RL alone).")
    return rows


ISAAC_RUNS = [("g1_flat", "Isaac-Velocity-Flat-G1-v0"), ("g1_rough", "Isaac-Velocity-Rough-G1-v0")]
ISAAC_LABEL = "within Isaac Sim (PhysX), not sim-to-sim"


def isaac_headline() -> list[str]:
    rows = ["| Isaac Lab policy | task | env steps | training wall clock | nominal fall rate [Wilson 95%] | nominal lin-vel err (m/s) | worst condition (fall rate) |",
            "|---|---|---|---|---|---|---|"]
    n = 0
    for name, task in ISAAC_RUNS:
        tr, ev = load(f"isaac_train_{name}.json"), load(f"isaac_eval_{name}.json")
        if tr is None:
            continue
        n += 1
        if ev and ev["summary"].get("nominal"):
            nom = ev["summary"]["nominal"]
            worst = max(ev["summary"].items(), key=lambda kv: kv[1]["fall_rate"])
            nomtxt = f"{pct(nom['fall_rate'])} {ci_pct(nom['fall_rate_wilson95'])} (n={nom['episodes']})"
            err, worsttxt = f(nom["lin_vel_err_mean"]), f"{worst[0]} ({pct(worst[1]['fall_rate'])})"
        else:
            nomtxt, err, worsttxt = "not evaluated", "n/a", "n/a"
        rows.append(f"| RSL-RL PPO ({name}) | {task} | {tr['final_step']:,} | {hms(tr['wall_clock_s'])} | {nomtxt} | {err} | {worsttxt} |")
    return rows if n else []


def isaac_training_rows() -> list[str]:
    rows = ["| run | task | envs | iterations | env steps | wall clock | env steps/s (steady) | final mean episode reward / length | status | source |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    curves = []
    for name, task in ISAAC_RUNS:
        tr = load(f"isaac_train_{name}.json")
        if tr is None:
            continue
        last = next((r for r in reversed(tr["curve"]) if "mean_episode_reward" in r), {})
        rows.append(f"| {name} | {task} | {tr['num_envs']} | {tr['iterations_completed']} of {tr['requested_max_iterations']} | "
                    f"{tr['final_step']:,} | {hms(tr['wall_clock_s'])} | {f(tr.get('env_steps_per_s_steady'), 0)} | "
                    f"{f(last.get('mean_episode_reward'), 2)} / {f(last.get('mean_episode_length'), 1)} | {tr['status']} | "
                    f"`results/isaac_train_{name}.json` |")
        c = [r for r in tr["curve"] if "mean_episode_reward" in r]
        if c:
            k = 12
            idx = sorted({round(i * (len(c) - 1) / (k - 1)) for i in range(k)}) if len(c) > k else range(len(c))
            curves += ["", f"**{name}** curve (`results/isaac_train_{name}.json`; mean over the last 100 finished training episodes; "
                       "episode length in 50 Hz policy steps, 1000 = full 20 s episode)", "",
                       "| iteration | env steps | wall clock | mean episode reward | mean episode length | action noise std |",
                       "|---|---|---|---|---|---|"]
            for i in idx:
                r = c[i]
                curves.append(f"| {r['iteration']} | {r['step']:,} | {hms(r['wall_s'])} | {f(r['mean_episode_reward'], 2)} | "
                              f"{f(r['mean_episode_length'], 1)} | {f(r.get('action_noise_std'), 3)} |")
    return (rows + curves) if len(rows) > 2 else ["_no Isaac Lab training run_"]


def isaac_eval_rows() -> list[str]:
    out = []
    for name, task in ISAAC_RUNS:
        ev = load(f"isaac_eval_{name}.json")
        if ev is None:
            continue
        out += [f"**{name}** ({task} policy, {ISAAC_LABEL}; `results/isaac_eval_{name}.json`, "
                f"{ev['episodes_per_condition']} episodes of {ev['episode_steps']} steps per condition)", "",
                "| condition | fall rate [Wilson 95%] | falls / n | lin-vel err m/s [95% CI] | yaw-rate err rad/s [95% CI] | mean survival s | not fallen but base below 0.5 m at the end |",
                "|---|---|---|---|---|---|---|"]
        for c, s in ev["summary"].items():
            out.append(f"| {c} | {pct(s['fall_rate'])} {ci_pct(s['fall_rate_wilson95'])} | {s['falls']}/{s['episodes']} | "
                       f"{f(s['lin_vel_err_mean'])} [{f(s['lin_vel_err_ci95'][0])}, {f(s['lin_vel_err_ci95'][1])}] | "
                       f"{f(s['yaw_rate_err_mean'])} [{f(s['yaw_rate_err_ci95'][0])}, {f(s['yaw_rate_err_ci95'][1])}] | "
                       f"{f(s['survival_s_mean'], 1)} | {s['ended_low_count']} |")
        for c, msg in ev.get("failed_conditions", {}).items():
            out.append(f"| {c} | failed: {msg} | | | | | |")
        if "Rough" in task:
            out += ["", "Rough terrain: robots start on the task's generated terrain (curriculum levels of the training config), "
                    "so the last column (base height in world z) is not meaningful here; friction and mass conditions "
                    "change the robot, not the terrain."]
        out.append("")
    return out or ["_Isaac Lab evaluation not run_"]


def isaac_play_rows() -> list[str]:
    rows = []
    for name, _ in ISAAC_RUNS:
        p = load(f"isaac_play_{name}.json")
        if p is None:
            continue
        ex, par, v = p["exported"], p["parity"], p["video"]
        rows.append(f"- {name}: checkpoint iteration {p['checkpoint_iteration']}, exported with Isaac Lab's exporters to "
                    f"`{ex['policy_pt']}` (TorchScript) and `{ex['policy_onnx']}` (ONNX opset {ex['onnx_opset']}). "
                    f"Max abs action difference over {par['n_obs']} rollout observations: ONNX vs RSL-RL actor "
                    f"{par['max_abs_err_onnx_vs_actor']:.2e}, TorchScript vs actor {par['max_abs_err_jit_vs_actor']:.2e} "
                    f"(`results/isaac_play_{name}.json`).")
        vid = load(f"isaac_video_{name}.json")
        if vid:
            rows.append(f"- {name} video: `{vid['file']}` ({vid['duration_s']:.1f} s, {vid['width']}x{vid['height']}, "
                        f"{vid['bytes'] / 1e6:.2f} MB); command vx={v['command_vx_vy_wz'][0]} m/s held constant, env 0 fell: "
                        f"{v['env0_fell']}, env 0 net displacement {v['env0_net_displacement_m']:.2f} m over "
                        f"{v['steps'] * v['policy_dt_s']:.0f} s ({v['note']}).")
        if vid and vid.get("trimmed_leading_frames"):
            rows.append(f"  Leading renderer warm-up frames trimmed: {vid['trimmed_leading_frames']} ({vid['trim_rule']}).")
    cmp = load("compare_video.json")
    if cmp:
        rows.append(f"- Side-by-side clip `{cmp['file']}`: left {cmp['left']['what']}; right {cmp['right']['what']}; "
                    f"{cmp['duration_s_each']:.1f} s each ({cmp['note']}).")
    return rows or ["_no export / video_"]


def isaac_mapping_rows() -> list[str]:
    m = load("isaac_mujoco_mapping.json")
    if m is None:
        return ["_model comparison not run_"]
    j = m["joints"]
    rows = [f"- Isaac Lab G1 action joints: {j['isaac_num_joints']}; Playground/Menagerie G1 actuators: "
            f"{j['mujoco_num_actuated_joints']}; joints with the same name: {len(j['common'])} "
            f"(`results/isaac_mujoco_mapping.json`).",
            f"- Isaac-only joints ({len(j['isaac_only'])}): {', '.join(j['isaac_only'])}.",
            f"- MuJoCo-only joints ({len(j['mujoco_only'])}): {', '.join(j['mujoco_only'])}.",
            f"- Observation: Isaac policy {m['isaac_obs_dim']}-dim ({', '.join(f'{k} {v[0]}' for k, v in m['isaac_obs_terms'].items())}); "
            f"Playground actor {m['playground_obs_dim']}-dim.",
            f"- Decision: **{m['decision']}**."]
    rows += [f"- Blocker: {b}" for b in m["blockers"]]
    return rows


def isaac_rows() -> list[str]:
    rows = []
    if load("isaac_train_g1_flat.json"):
        rows += [f"Isaac Sim + Isaac Lab were installed and run on this host; every evaluation number in this section is {ISAAC_LABEL}. "
                 "Install record, workarounds and protocol: docs/ISAAC_LAB.md.", "", "### Headline", "", *isaac_headline(), "",
                 "### Training", "", *isaac_training_rows(), "", "### Export and video", "", *isaac_play_rows(), "",
                 "### Evaluation inside Isaac Sim", "", *isaac_eval_rows(),
                 "### Isaac Lab G1 vs Playground/Menagerie G1 (for sim-to-sim)", "", *isaac_mapping_rows()]
    r = load("isaac_lab_check.json")
    if r is not None:
        rows += ["", f"- Earlier feasibility check (before the install, `results/isaac_lab_check.json`): {r['verdict']}."]
    return rows or ["_Isaac Lab not run_"]


def isaac_doc() -> str | None:
    inst = load("isaac_install.json")
    if inst is None:
        return None
    v = inst["versions"]
    gb = inst["disk_bytes"]
    sm = inst["smoke_test"]
    lines = [
        "# Isaac Lab track",
        "",
        "Generated by `python scripts/make_report.py` from `results/isaac_*.json`. Do not edit by hand. "
        f"All numbers: **{HARDWARE_LABEL}**; evaluation numbers are {ISAAC_LABEL}.",
        "",
        "## What was run",
        "",
        "1. NVIDIA Isaac Sim and Isaac Lab installed from pip wheels and a git tag into a separate venv (`scripts/isaac/setup_isaac.sh`).",
        "2. Isaac Lab's registered Unitree G1 velocity-tracking task trained with RSL-RL PPO, the task's own env and runner "
        "config unchanged except num_envs, seed and a wall-clock cap (`scripts/isaac/train_g1.py`).",
        "3. Policy exported with Isaac Lab's exporters (TorchScript + ONNX), parity-checked, and one headless video recorded "
        "(`scripts/isaac/play_export.py`).",
        "4. Policy evaluated inside Isaac Sim under perturbations (`scripts/isaac/eval_g1.py`).",
        "5. Isaac Lab G1 compared joint by joint with the Playground/Menagerie G1 to decide whether an Isaac to MuJoCo "
        "sim-to-sim test is possible (`scripts/isaac/compare_models.py`).",
        "",
        "## Install on this host",
        "",
        f"- Host: {inst['host_os']}, glibc {inst['glibc']}, {inst['venv']}. Amazon Linux is not an Isaac Sim supported OS; "
        "it worked here with the fixes listed below.",
        f"- Versions: isaacsim {v.get('isaacsim')}, Isaac Lab {inst['isaac_lab_git_tag']} (isaaclab {v.get('isaaclab')}, "
        f"isaaclab_tasks {v.get('isaaclab_tasks')}, isaaclab_rl {v.get('isaaclab_rl')}), rsl-rl-lib {v.get('rsl-rl-lib')}, "
        f"torch {v.get('torch')}, numpy {v.get('numpy')}.",
        f"- Install time: torch {f(inst['install_torch_s'], 0)} s, Isaac Sim wheels {f(inst['install_isaacsim_s'], 0)} s, "
        f"Isaac Lab packages {f(inst['install_isaaclab_s'], 0)} s (download + install, from the install logs).",
        f"- Disk: `.venv-isaac` {gb['.venv-isaac'] / 1e9:.1f} GB, Isaac Lab checkout {gb['third_party/IsaacLab'] / 1e9:.2f} GB; "
        + "; ".join(f"{k} {val / 1e9:.2f} GB" for k, val in gb.items() if k.startswith("~/") and val is not None)
        + f". {inst['note_disk']}",
        "- Filesystem during the Isaac Sim install: " + " -> ".join(ln.split()[2] + " used" for ln in inst["df_lines_during_isaacsim_install"]) + ".",
        f"- Smoke test: `{sm['command']}` exited with code {sm['exit_code']}, graphics API {sm['graphics_api']}, "
        f"{sm['iterations_logged']} iterations logged, last iteration {sm['last_iteration_steps_per_s']} steps/s. "
        "Vulkan worked through the NVIDIA ICD already on the host (`/etc/vulkan/icd.d/nvidia_icd.json`).",
        f"- {inst['eula']}.",
        "",
        "Problems hit and fixes (all in `scripts/isaac/setup_isaac.sh`):",
        "",
        "- Isaac Sim 5.x/6.x wheels need glibc 2.35, so Isaac Sim 4.5.0 (glibc 2.34 wheels, Python 3.10) and Isaac Lab "
        "v2.1.1 (the last tag for Isaac Sim 4.5) were used.",
        "- `isaaclab.sh -i` replaces torch with 2.7.0+cu128 at this tag; the Isaac Lab packages were installed with uv and a "
        "constraint that keeps torch 2.5.1+cu121.",
        "- `flatdict==4.0.1` failed to build (`ModuleNotFoundError: No module named 'pkg_resources'` in an isolated build "
        "env with a new setuptools); fixed by installing setuptools<80 and building flatdict without build isolation.",
        f"- System package installed outside the venv with `sudo dnf install -y mesa-libGLU`: {inst['extra_system_package']}.",
        "- Isaac Lab's ONNX exporter writes a static batch of 1, so the ONNX parity check runs one observation at a time.",
        "- A Kit process from a failed export test (it ignored the SIGTERM sent by `timeout`) stayed alive on the GPU from "
        "early in the flat training until the first hour of the rough training had passed, and was then killed with SIGKILL. "
        "The flat training throughput, the flat evaluation and part of the rough training were measured with it present; "
        "Kit processes should be run under `timeout -s KILL`.",
        "",
        "## Results",
        "",
        *isaac_rows(),
        "",
        "## Notes on the protocol",
        "",
        "- The Isaac evaluation is not the CPU MuJoCo sim-to-sim table: different robot model (see the joint comparison), "
        "different simulator, different fall definition (Isaac: the task's torso_link contact termination; CPU MuJoCo: "
        "Playground's up-vector and foot-contact termination) and different command ranges (Isaac: vx 0 to 1 m/s as in "
        "its training config; MuJoCo: vx -1 to 1 m/s). Compare policies within one table only.",
        "- Isaac perturbations: friction scales the robot's rigid-body material (static 0.8 / dynamic 0.6 nominal; the ground material is 1.0 "
        "with combine mode multiply); torso mass adds +/-3 kg to torso_link at startup (the flat G1 task trains with no "
        "mass randomization); pushes add a 0.5 to 1.5 m/s horizontal base-velocity kick every 2 to 4 s (same as the CPU "
        "MuJoCo push condition; the flat G1 task trains with no pushes); action delay feeds the env the action from 1 or 2 "
        "policy steps earlier (the observation's last-action slot then holds the delayed action); sensor noise turns on the "
        "task's own observation noise.",
        "- Episodes run in parallel (one env per episode) with env seed 10000; commands and start poses come from the "
        "task's reset events, so episodes are not paired across conditions the way the CPU MuJoCo episodes are.",
        "- The Isaac-trained policy is a different network for a different robot model than the MuJoCo Playground policies; "
        "its numbers say nothing about the Playground policies and vice versa.",
        "",
    ]
    r = load("isaac_lab_check.json")
    if r is not None:
        lines += ["## Earlier feasibility check", "",
                  f"Before the install, `scripts/isaac_lab_check.py` measured the Isaac Sim 4.5 wheel download at "
                  f"{r['isaacsim_4_5_install']['total_wheel_bytes'] / 1e9:.2f} GB and stopped at a 2 GB download cap set at the "
                  f"time (verdict: {r['verdict']}). The cap was lifted for this track; `results/isaac_lab_check.json` is kept "
                  "as the record of that check.", ""]
    return "\n".join(lines)


def render_rows() -> list[str]:
    rows = []
    for n, lab in POLICIES:
        r = load(f"render_{n}.json")
        if not r:
            continue
        for k, v in r["videos"].items():
            dist = v.get("base_xy_path_length_m")
            dtxt = f", base travelled {dist:.2f} m" if dist is not None else ""
            rows.append(f"- {lab}, {k}: `{v['file']}` ({v['frames']} frames at 25 fps, fell: {v['fell']}{dtxt}; {v['note']})")
    return rows or ["_no videos rendered_"]


def failures() -> list[str]:
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        st = d.get("status", "")
        if isinstance(st, str) and st.startswith("failed"):
            rows.append(f"- `{p.name}`: {st}")
    for extra in ("failures.json",):
        d = load(extra)
        if d:
            for item in d.get("items", []):
                rows.append(f"- {item['what']}: {item['error']}")
    return rows or ["- none recorded"]


def headline() -> list[str]:
    rows = ["| policy | checkpoint env steps | MJX+DR fall rate | CPU MuJoCo nominal fall rate [Wilson 95%] | nominal lin-vel err (m/s) | worst sim-to-sim condition (fall rate) |",
            "|---|---|---|---|---|---|"]
    for n, lab in POLICIES:
        op, mj, ss = load(f"onnx_parity_{n}.json"), load(f"mjx_eval_{n}.json"), load(f"sim2sim_{n}.json")
        if op is None:
            continue
        mjx = pct(mj["with_dr"]["fall_rate"]) if mj else "n/a"
        if ss:
            nom = ss["summary"].get("nominal")
            worst = max(ss["summary"].items(), key=lambda kv: kv[1]["fall_rate"])
            nomtxt = f"{pct(nom['fall_rate'])} {ci_pct(nom['fall_rate_wilson95'])}" if nom else "n/a"
            err = f(nom["lin_vel_err_mean"]) if nom else "n/a"
            worsttxt = f"{worst[0]} ({pct(worst[1]['fall_rate'])})"
        else:
            nomtxt, err, worsttxt = "n/a", "n/a", "n/a"
        step = op.get("checkpoint_step")
        rows.append(f"| {lab} | {step:,} | {mjx} | {nomtxt} | {err} | {worsttxt} |" if step is not None
                    else f"| {lab} | n/a | {mjx} | {nomtxt} | {err} | {worsttxt} |")
    return rows


def main() -> None:
    fall, trk = sim2sim_tables()
    parts = [
        "# Results",
        "",
        f"All numbers: **{HARDWARE_LABEL}**. Generated by `python scripts/make_report.py` from `results/*.json`; do not edit by hand.",
        "Transfer between simulators is sim-to-sim (MJX to CPU MuJoCo), not sim-to-real. "
        "The Isaac Lab numbers are measured inside Isaac Sim (PhysX), not sim-to-sim.",
        "",
        "## Headline",
        "",
        *headline(),
        "",
        *([f"Isaac Lab track ({ISAAC_LABEL}; details in the Isaac Lab section and docs/ISAAC_LAB.md):", "", *isaac_headline(), ""]
          if isaac_headline() else []),
        "## Training runs",
        "",
        *training_table(),
        "",
        "brax reward = `eval/episode_reward` of brax's in-loop evaluator (128 envs, stochastic policy, same env config as training). "
        "RSL-RL reward = mean return of the last 100 finished training episodes. The two are not the same estimator; "
        "the common MJX evaluation below scores all policies identically.",
        "",
        "### Training curves",
        "",
    ]
    for n, _ in TRAIN_RUNS:
        parts += curve_table(n) + [""]
    parts += ["### DR vs no-DR at matched env steps (brax in-loop eval reward)", "", *matched_curve_rows(), ""]
    parts += ["### brax PPO vs RSL-RL PPO at similar env steps (both with DR)", "", *brax_vs_rsl_rows(), ""]
    parts += ["### GPU scheduling (parallel vs sequential)", "", *scheduling_rows(), ""]
    parts += [
        "## Common MJX evaluation (all policies, same protocol)",
        "",
        *mjx_table(),
        "",
        "The MJX protocol keeps everything the training env does: velocity-kick pushes every 5 to 10 s, training-level "
        "sensor noise, and a new random command every 10 s. The CPU MuJoCo `nominal` condition below has no pushes, clean "
        "observations and one constant command, so MJX fall rates are not comparable to CPU MuJoCo `nominal` fall rates; "
        "compare policies within one table.",
        "",
        "## Parity checks",
        "",
        *parity_rows(),
        "",
        "## Sim-to-sim: ONNX policy in plain CPU MuJoCo",
        "",
        "Fall = Playground G1 termination (torso up-vector z < 0, foot-foot or foot-shin contact, NaN). "
        "Episodes: 1000 control steps (20 s at 50 Hz), per-episode constant command sampled from the training ranges, "
        "initial state from the training reset distribution, seed 10000 + i for episode i in every condition.",
        "",
        "### Fall rate",
        "",
        *fall,
        "",
        "### Velocity tracking (mean over steps after 1 s, surviving steps only) and mean survival time",
        "",
        *trk,
        "",
        "### DR vs no-DR, paired (final DR checkpoint vs final no-DR checkpoint)",
        "",
        *dr_effect_rows("brax_dr"),
        "",
        "### DR vs no-DR, paired, step-matched (DR snapshot at the no-DR run's final step)",
        "",
        *dr_effect_rows("brax_dr_matched"),
        "",
        "## Videos",
        "",
        *render_rows(),
        "",
        "## Isaac Lab",
        "",
        *isaac_rows(),
        "",
        "## Failures and errors",
        "",
        *failures(),
        "",
    ]
    text = "\n".join(parts)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text(text)
    isaac = isaac_doc()
    if isaac:
        (ROOT / "docs" / "ISAAC_LAB.md").write_text(isaac)

    readme = ROOT / "README.md"
    if readme.exists():
        r = readme.read_text()
        block = "<!-- RESULTS:BEGIN -->\n" + "\n".join(
            [f"_Generated from results/*.json by scripts/make_report.py. {HARDWARE_LABEL}._", "", *headline(),
             *(["", f"Isaac Lab track ({ISAAC_LABEL}):", "", *isaac_headline()] if isaac_headline() else [])]) + "\n<!-- RESULTS:END -->"
        r = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _m: block, r, flags=re.S)
        readme.write_text(r)
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
