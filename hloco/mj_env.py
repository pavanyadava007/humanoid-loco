"""Plain CPU MuJoCo re-implementation of the Playground G1 joystick task (policy side only).

It rebuilds exactly the 103-dim "state" observation that the brax/RSL-RL actor sees in
mujoco_playground/_src/locomotion/g1/joystick.py, including two quirks of that code:
  * the "last_act" slot holds the action from two control steps ago at obs time
    (info["last_act"] is updated after the observation is built), and
  * the gait phase in the observation is the phase before the per-step increment.
Rewards are not reimplemented; only termination, observation and actuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

FEET_FLOOR_SENSORS = ("left_foot_floor_found", "right_foot_floor_found")
TERMINATION_SENSORS = (
    "right_foot_left_foot_found",
    "left_foot_right_shin_found",
    "right_foot_left_shin_found",
)
TORSO_BODY = "torso_link"


def load_model() -> mujoco.MjModel:
    """Load the same G1 flat-terrain MJCF (and asset set) the Playground env uses."""
    from etils import epath
    from mujoco_playground._src.locomotion.g1 import base as g1_base
    from mujoco_playground._src.locomotion.g1 import g1_constants as consts

    xml = epath.Path(consts.task_to_xml("flat_terrain")).read_text()
    model = mujoco.MjModel.from_xml_string(xml, assets=g1_base.get_assets())
    model.opt.timestep = 0.002
    return model


@dataclass
class Perturbation:
    """Test-time perturbation of the nominal model / loop."""

    name: str = "nominal"
    friction_scale: float = 1.0
    torso_mass_delta_kg: float = 0.0
    action_delay_steps: int = 0
    push: bool = False
    push_interval_s: tuple[float, float] = (2.0, 4.0)
    push_magnitude: tuple[float, float] = (0.5, 1.5)
    solver: str = "same"  # "same" = MJX-matching options; "accurate" = Newton 100 iters, implicitfast
    obs_noise: float = 0.0  # multiplies the training noise scales (0 = clean obs)


@dataclass
class G1MjEnv:
    model: mujoco.MjModel
    pert: Perturbation = field(default_factory=Perturbation)
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    action_scale: float = 0.5

    def __post_init__(self) -> None:
        m = self.model
        self.n_substeps = int(round(self.ctrl_dt / self.sim_dt))
        self.data = mujoco.MjData(m)
        key = m.keyframe("knees_bent")
        self.init_q = key.qpos.copy()
        self.default_pose = key.qpos[7:].copy()
        self.nu = m.nu
        self.pelvis_imu_site = m.site("imu_in_pelvis").id
        self.torso_id = m.body(TORSO_BODY).id
        self._sadr = {
            n: m.sensor_adr[m.sensor(n).id]
            for n in (*FEET_FLOOR_SENSORS, *TERMINATION_SENSORS)
        }
        self.nominal = {
            "pair_friction": m.pair_friction.copy(),
            "body_mass": m.body_mass.copy(),
            "opt": (m.opt.iterations, m.opt.ls_iterations, m.opt.integrator, m.opt.solver),
        }
        self.noise_scales = {
            "joint_pos": 0.03,
            "joint_vel": 1.5,
            "gravity": 0.05,
            "linvel": 0.1,
            "gyro": 0.2,
        }
        self.apply_perturbation(self.pert)

    # ------------------------------------------------------------------ model edits
    def apply_perturbation(self, pert: Perturbation) -> None:
        m = self.model
        m.pair_friction[:] = self.nominal["pair_friction"]
        m.body_mass[:] = self.nominal["body_mass"]
        it, ls, integ, solver = self.nominal["opt"]
        m.opt.iterations, m.opt.ls_iterations, m.opt.integrator, m.opt.solver = it, ls, integ, solver
        # Floor/foot pairs are the first two <pair> entries (same indices the Playground
        # randomizer edits: pair_friction[0:2, 0:2]).
        m.pair_friction[0:2, 0:2] = self.nominal["pair_friction"][0:2, 0:2] * pert.friction_scale
        m.body_mass[self.torso_id] = self.nominal["body_mass"][self.torso_id] + pert.torso_mass_delta_kg
        if pert.solver == "accurate":
            m.opt.iterations = 100
            m.opt.ls_iterations = 50
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.pert = pert

    # ------------------------------------------------------------------ sensors / obs
    def sensor(self, name: str) -> np.ndarray:
        return self.data.sensor(name).data.copy()

    def gravity_pelvis(self) -> np.ndarray:
        xmat = self.data.site_xmat[self.pelvis_imu_site].reshape(3, 3)
        return xmat.T @ np.array([0.0, 0.0, -1.0])

    def feet_contact(self) -> np.ndarray:
        return np.array([self.data.sensordata[self._sadr[n]] > 0 for n in FEET_FLOOR_SENSORS])

    def obs(self, command: np.ndarray, last_act: np.ndarray, phase: np.ndarray,
            rng: np.random.Generator | None = None) -> np.ndarray:
        d = self.data
        parts = {
            "linvel": self.sensor("local_linvel_pelvis"),
            "gyro": self.sensor("gyro_pelvis"),
            "gravity": self.gravity_pelvis(),
            "joint_pos": d.qpos[7:].copy(),
            "joint_vel": d.qvel[6:].copy(),
        }
        if self.pert.obs_noise > 0 and rng is not None:
            for k, v in parts.items():
                parts[k] = v + (2 * rng.random(v.shape) - 1) * self.pert.obs_noise * self.noise_scales[k]
        return np.concatenate([
            parts["linvel"],
            parts["gyro"],
            parts["gravity"],
            command,
            parts["joint_pos"] - self.default_pose,
            parts["joint_vel"],
            last_act,
            np.cos(phase),
            np.sin(phase),
        ]).astype(np.float32)

    def terminated(self) -> bool:
        d = self.data
        up_z = self.sensor("upvector_torso")[-1]
        bad = up_z < 0.0
        for n in TERMINATION_SENSORS:
            bad |= d.sensordata[self._sadr[n]] > 0
        bad |= bool(np.isnan(d.qpos).any() or np.isnan(d.qvel).any())
        return bool(bad)

    # ------------------------------------------------------------------ dynamics
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray | None = None) -> None:
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        if ctrl is not None:
            d.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, d)

    def reset_random(self, rng: np.random.Generator) -> None:
        """Same initial-state distribution as Joystick.reset (numpy RNG instead of JAX RNG)."""
        qpos = self.init_q.copy()
        qpos[0:2] += rng.uniform(-0.5, 0.5, size=2)
        yaw = rng.uniform(-3.14, 3.14)
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw)
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, qpos[3:7], quat)
        qpos[3:7] = new_quat
        qpos[7:] = qpos[7:] * rng.uniform(0.5, 1.5, size=self.nu)
        qvel = np.zeros(self.model.nv)
        qvel[0:6] = rng.uniform(-0.5, 0.5, size=6)
        self.set_state(qpos, qvel, ctrl=qpos[7:])

    def step_ctrl(self, motor_targets: np.ndarray) -> None:
        d = self.data
        d.ctrl[:] = motor_targets
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, d)

    def motor_targets(self, action: np.ndarray) -> np.ndarray:
        return self.default_pose + action * self.action_scale
