"""Compare the Isaac Lab G1 (from results/isaac_play_<name>.json) with the MuJoCo Playground / Menagerie G1.

Decides whether an exact Isaac Lab to CPU MuJoCo sim-to-sim evaluation is possible: it needs a one-to-one
map between the Isaac policy's action joints and actuated MuJoCo joints and every observation term.
Runs in the main venv (.venv). Writes results/isaac_mujoco_mapping.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hloco.common import RESULTS, read_json, write_json  # noqa: E402

PLAYGROUND_OBS_DIM = 103


def compare(isaac_joints: list[str], mj_joints: list[str]) -> dict:
    """Name-level comparison of two joint lists."""
    si, sm = set(isaac_joints), set(mj_joints)
    return {
        "isaac_num_joints": len(isaac_joints),
        "mujoco_num_actuated_joints": len(mj_joints),
        "common": sorted(si & sm),
        "isaac_only": sorted(si - sm),
        "mujoco_only": sorted(sm - si),
        "same_order_for_common": [j for j in isaac_joints if j in sm] == [j for j in mj_joints if j in si],
        "one_to_one": si == sm and len(isaac_joints) == len(mj_joints),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="g1_flat")
    args = ap.parse_args()
    import mujoco

    from hloco.mj_env import load_model

    play = read_json(RESULTS / f"isaac_play_{args.name}.json")
    model = load_model()
    mj_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0]))
                 for a in range(model.nu)]
    kp = {j: float(model.actuator_gainprm[a, 0]) for a, j in enumerate(mj_joints)}
    cmp = compare(play["model"]["joint_names"], mj_joints)
    obs_terms = play["model"]["obs_terms"]
    isaac_obs_dim = sum(int(v[0]) for v in obs_terms.values())
    blockers = []
    if not cmp["one_to_one"]:
        blockers.append(
            f"joint sets differ: Isaac Lab G1 has {cmp['isaac_num_joints']} action joints, the Playground/Menagerie G1 has "
            f"{cmp['mujoco_num_actuated_joints']} actuators; {len(cmp['isaac_only'])} Isaac joints have no MuJoCo "
            f"counterpart and {len(cmp['mujoco_only'])} MuJoCo joints have no Isaac counterpart")
    if isaac_obs_dim != PLAYGROUND_OBS_DIM:
        blockers.append(f"observation layouts differ: Isaac policy obs {isaac_obs_dim}-dim ({', '.join(obs_terms)}), "
                        f"Playground actor obs {PLAYGROUND_OBS_DIM}-dim (29-joint position, velocity and last-action blocks "
                        "and a gait-phase term the Isaac task does not have)")
    write_json(RESULTS / "isaac_mujoco_mapping.json", {
        "isaac_source": f"results/isaac_play_{args.name}.json (robot articulation of Isaac-Velocity-Flat-G1-v0, g1_minimal.usd)",
        "mujoco_source": "mujoco_playground G1 flat_terrain MJCF (Menagerie unitree_g1), actuator list",
        "joints": cmp,
        "mujoco_actuator_kp": kp,
        "isaac_obs_terms": obs_terms,
        "isaac_obs_dim": isaac_obs_dim,
        "playground_obs_dim": PLAYGROUND_OBS_DIM,
        "exact_mapping_possible": not blockers,
        "blockers": blockers,
        "decision": "sim-to-sim not attempted" if blockers else "exact mapping exists",
    })
    print("wrote", RESULTS / "isaac_mujoco_mapping.json", "blockers:", len(blockers))


if __name__ == "__main__":
    main()
