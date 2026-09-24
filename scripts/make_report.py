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

POLICIES = [("brax_dr", "brax PPO + DR"), ("brax_nodr", "brax PPO, no DR"), ("rsl_dr", "RSL-RL PPO + DR")]


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
    for name, label in POLICIES:
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


def dr_effect_rows() -> list[str]:
    a, b = load("sim2sim_brax_dr.json"), load("sim2sim_brax_nodr.json")
    if not a or not b:
        return ["_DR vs no-DR comparison needs both sim2sim_brax_dr.json and sim2sim_brax_nodr.json_"]
    rows = ["| condition | falls DR | falls no-DR | paired episodes: only DR fell / only no-DR fell |", "|---|---|---|---|"]
    for c, s in a["summary"].items():
        if c not in b["summary"]:
            continue
        ea, eb = a["episodes"][c], b["episodes"][c]
        only_a = sum(x["fell"] and not y["fell"] for x, y in zip(ea, eb, strict=True))
        only_b = sum(y["fell"] and not x["fell"] for x, y in zip(ea, eb, strict=True))
        rows.append(f"| {c} | {s['falls']}/{s['episodes']} | {b['summary'][c]['falls']}/{b['summary'][c]['episodes']} | {only_a} / {only_b} |")
    return rows + ["", "Same seeds (initial state, command, pushes) for both policies, so episodes are paired."]


def isaac_rows() -> list[str]:
    r = load("isaac_lab_check.json")
    if r is None:
        return ["_feasibility check not run_"]
    return [f"- Verdict: **{r['verdict']}** (`results/isaac_lab_check.json`, details in docs/ISAAC_LAB.md)."]


def isaac_doc() -> str | None:
    r = load("isaac_lab_check.json")
    if r is None:
        return None
    lines = [
        "# Isaac Lab feasibility check",
        "",
        "Generated by `python scripts/make_report.py` from `results/isaac_lab_check.json` "
        "(written by `python scripts/isaac_lab_check.py`). Do not edit by hand.",
        "",
        "Isaac Lab runs on top of NVIDIA Isaac Sim. The check only reads the NVIDIA pip index "
        f"({r['index']}) and sends HTTP HEAD requests; it downloads and installs nothing "
        f"(wheel bytes downloaded: {r['wheel_bytes_downloaded']}).",
        "",
        "## Host",
        "",
        f"- OS: {r['host_os']}",
        f"- glibc: {r['host_glibc']}",
        f"- Python (project venv): {r['host_python']}",
        f"- GPU / driver: {r['gpu_and_driver']}",
        f"- Free disk at check time: {r['free_disk_gb']} GB",
        "",
        "## Isaac Sim wheels on the index (Linux x86_64)",
        "",
        "| package | glibc tags | python tags | newest wheel installable with host glibc | its size (HEAD) | newest wheel overall | its size (HEAD) |",
        "|---|---|---|---|---|---|---|",
    ]
    def mb(x):
        return "n/a" if x is None else f"{x / 1e6:.1f} MB"
    for pkg, info in r["packages"].items():
        if "error" in info:
            lines.append(f"| {pkg} | error: {info['error']} | | | | | |")
            continue
        lines.append(f"| {pkg} | {', '.join(info['manylinux_glibc_tags'])} | {', '.join(info['python_tags'])} | "
                     f"{info['newest_host_installable_wheel']} | {mb(info['newest_host_installable_wheel_bytes_HEAD'])} | "
                     f"{info['latest_wheel']} | {mb(info['latest_wheel_bytes_HEAD'])} |")
    lines += ["", f"## Verdict: {r['verdict']}", ""]
    lines += [f"- Blocker: {b}" for b in r["blockers"]] or ["- No blocker found."]
    lines += [""] + [f"- Note: {n}" for n in r.get("notes", [])]
    lines += ["", "Consequence for this repo: no Isaac Lab training or evaluation was run. All results come from "
              "MuJoCo Playground (MJX) training and plain CPU MuJoCo evaluation.", ""]
    return "\n".join(lines)


def render_rows() -> list[str]:
    rows = []
    for n, lab in POLICIES:
        r = load(f"render_{n}.json")
        if not r:
            continue
        for k, v in r["videos"].items():
            rows.append(f"- {lab}, {k}: `{v['file']}` ({v['frames']} frames, fell: {v['fell']}; {v['note']})")
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
    rows = ["| policy | train env steps | MJX+DR fall rate | CPU MuJoCo nominal fall rate [Wilson 95%] | nominal lin-vel err (m/s) | worst sim-to-sim condition (fall rate) |",
            "|---|---|---|---|---|---|"]
    for n, lab in POLICIES:
        tr, mj, ss = load(f"train_{n}.json"), load(f"mjx_eval_{n}.json"), load(f"sim2sim_{n}.json")
        if tr is None:
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
        rows.append(f"| {lab} | {tr['final_step']:,} | {mjx} | {nomtxt} | {err} | {worsttxt} |")
    return rows


def main() -> None:
    fall, trk = sim2sim_tables()
    parts = [
        "# Results",
        "",
        f"All numbers: **{HARDWARE_LABEL}**. Generated by `python scripts/make_report.py` from `results/*.json`; do not edit by hand.",
        "Transfer between simulators is sim-to-sim (MJX to CPU MuJoCo), not sim-to-real.",
        "",
        "## Headline",
        "",
        *headline(),
        "",
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
    for n, _ in POLICIES:
        parts += curve_table(n) + [""]
    parts += [
        "## Common MJX evaluation (all policies, same protocol)",
        "",
        *mjx_table(),
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
        "### DR vs no-DR, paired",
        "",
        *dr_effect_rows(),
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
            [f"_Generated from results/*.json by scripts/make_report.py. {HARDWARE_LABEL}._", "", *headline()]) + "\n<!-- RESULTS:END -->"
        r = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _m: block, r, flags=re.S)
        readme.write_text(r)
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
