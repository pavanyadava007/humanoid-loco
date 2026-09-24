"""Export a trained Isaac Lab G1 policy (policy.pt TorchScript + policy.onnx) and record a headless video.

Runs in .venv-isaac. Uses Isaac Lab's own exporters (isaaclab_rl.rsl_rl.export_policy_as_jit / _onnx),
checks ONNX vs TorchScript vs the RSL-RL actor on observations from the rollout, and records one
video of env 0 with a fixed forward command through gymnasium's RecordVideo wrapper (Isaac Lab --video path).
Writes results/isaac_play_<name>.json and copies the exported files to checkpoints/isaac_<name>/.

Example:
    OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/isaac/play_export.py --headless --enable_cameras \
        --name g1_flat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-Play-v0")
parser.add_argument("--name", default="g1_flat")
parser.add_argument("--checkpoint", default=None, help="default: last model_*.pt in logs/isaac/runs/<name>")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--video_steps", type=int, default=500, help="policy steps recorded (50 Hz)")
parser.add_argument("--command", type=float, nargs=3, default=[0.8, 0.0, 0.0], help="vx vy wz, held constant")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import shutil  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hloco.common import CKPT, MEDIA, RESULTS, write_json  # noqa: E402


def last_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return ckpts[-1]


def main() -> None:
    run_dir = ROOT / "logs" / "isaac" / "runs" / args.name
    ckpt = Path(args.checkpoint) if args.checkpoint else last_checkpoint(run_dir)

    env_cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    vx, vy, wz = args.command
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (wz, wz)
    cmd.debug_vis = False
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.video_steps * 0.02 + 1.0)
    # camera follows the robot of env 0
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (2.2, -2.2, 1.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.viewer.resolution = (960, 540)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    video_dir = ROOT / "logs" / "isaac" / "video" / args.name
    if video_dir.exists():
        shutil.rmtree(video_dir)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(video_dir), step_trigger=lambda s: s == 0,
                                   video_length=args.video_steps, disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))
    ckpt_iter = runner.current_learning_iteration
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = run_dir / "exported"
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=str(export_dir), filename="policy.pt")
    export_policy_as_onnx(runner.alg.policy, normalizer=runner.obs_normalizer, path=str(export_dir), filename="policy.onnx")
    jit = torch.jit.load(str(export_dir / "policy.pt"), map_location=env.unwrapped.device)
    onnx_model = onnx.load(str(export_dir / "policy.onnx"))
    sess = ort.InferenceSession(str(export_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    dev = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    obs, _ = env.get_observations()
    obs_log, act_log = [], []
    start_xy = robot.data.root_pos_w[0, :2].clone()
    path_len, fell, prev_xy = 0.0, False, start_xy.clone()
    heights = []
    for t in range(args.video_steps):
        with torch.inference_mode():
            act = policy(obs)
        if t % 10 == 0:
            obs_log.append(obs.detach().cpu().numpy().copy())
            act_log.append(act.detach().cpu().numpy().copy())
        obs, _, dones, _ = env.step(act)
        if bool(env.unwrapped.reset_terminated[0]):
            fell = True
        xy = robot.data.root_pos_w[0, :2]
        if not bool(dones[0]):
            path_len += float(torch.linalg.norm(xy - prev_xy))
        prev_xy = xy.clone()
        heights.append(float(robot.data.root_pos_w[0, 2]))
    net_disp = float(torch.linalg.norm(robot.data.root_pos_w[0, :2] - start_xy))
    joint_names = list(robot.data.joint_names)
    obs_terms = {
        name: list(shape) for name, shape in zip(
            env.unwrapped.observation_manager.active_terms["policy"],
            env.unwrapped.observation_manager.group_obs_term_dim["policy"], strict=True)
    }
    default_joint_pos = robot.data.default_joint_pos[0].cpu().tolist()
    env.close()

    obs_arr = np.concatenate(obs_log).astype(np.float32)
    ref = np.concatenate(act_log)
    with torch.inference_mode():
        jit_out = jit(torch.as_tensor(obs_arr, device=dev)).cpu().numpy()
    onnx_out = np.concatenate([sess.run(None, {in_name: o[None]})[0] for o in obs_arr])  # static batch 1

    vids = sorted(video_dir.glob("*.mp4"))
    out_ck = CKPT / f"isaac_{args.name}"
    out_ck.mkdir(parents=True, exist_ok=True)
    shutil.copy2(export_dir / "policy.pt", out_ck / "policy.pt")
    shutil.copy2(export_dir / "policy.onnx", out_ck / "policy.onnx")
    shutil.copy2(ckpt, out_ck / "model.pt")
    for p in ("env.yaml", "agent.yaml"):
        if (run_dir / "params" / p).exists():
            shutil.copy2(run_dir / "params" / p, out_ck / p)
    raw_video = None
    if vids:
        MEDIA.mkdir(exist_ok=True)
        raw_video = str(vids[0].relative_to(ROOT))

    write_json(RESULTS / f"isaac_play_{args.name}.json", {
        "task": args.task,
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "checkpoint_iteration": ckpt_iter,
        "exported": {"policy_pt": f"checkpoints/isaac_{args.name}/policy.pt",
                     "policy_onnx": f"checkpoints/isaac_{args.name}/policy.onnx",
                     "onnx_opset": int(onnx_model.opset_import[0].version),
                     "exporter": "isaaclab_rl.rsl_rl.export_policy_as_jit / export_policy_as_onnx (Isaac Lab)"},
        "parity": {
            "n_obs": int(obs_arr.shape[0]),
            "max_abs_err_onnx_vs_actor": float(np.abs(onnx_out - ref).max()),
            "max_abs_err_jit_vs_actor": float(np.abs(jit_out - ref).max()),
            "reference": "RSL-RL actor mean (runner.get_inference_policy) on observations from the rollout",
        },
        "video": {
            "raw_file": raw_video,
            "steps": args.video_steps,
            "policy_dt_s": 0.02,
            "command_vx_vy_wz": args.command,
            "env0_fell": fell,
            "env0_path_length_m": path_len,
            "env0_net_displacement_m": net_disp,
            "env0_base_height_min_m": min(heights),
            "note": "Isaac Lab Play config: no pushes, no observation noise; camera follows env 0",
        },
        "model": {"joint_names": joint_names, "num_joints": len(joint_names), "obs_terms": obs_terms,
                  "default_joint_pos": default_joint_pos},
    })
    print("wrote", RESULTS / f"isaac_play_{args.name}.json")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
