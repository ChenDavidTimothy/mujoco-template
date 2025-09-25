from __future__ import annotations

import sys

import numpy as np

import mujoco_template as mt
from pendulum_common import (
    initialize_state as seed_pendulum,
    make_env as make_pendulum_env,
    make_tip_probes,
    resolve_pendulum_columns,
)
from pendulum_config import CONFIG


class PendulumPDController:
    """Simple PD torque controller that stabilizes the pendulum upright."""

    def __init__(self, kp: float, kd: float, target: float) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self.kp = float(kp)
        self.kd = float(kd)
        self.target = float(target)

    def prepare(self, model: mt.mj.MjModel, data: mt.mj.MjData) -> None:
        if model.nu != 1:
            raise mt.CompatibilityError("PendulumPDController expects a single actuator.")

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, t: float) -> None:
        angle = float(data.qpos[0])
        velocity = float(data.qvel[0])
        torque = -self.kp * (angle - self.target) - self.kd * velocity
        if hasattr(model, "actuator_forcerange") and model.actuator_forcerange.size >= 2:
            bounds = np.asarray(model.actuator_forcerange)[0]
            torque = float(np.clip(torque, bounds[0], bounds[1]))
        data.ctrl[0] = torque


def build_env() -> mt.Env:
    ctrl_cfg = CONFIG.controller
    controller = PendulumPDController(
        kp=ctrl_cfg.kp,
        kd=ctrl_cfg.kd,
        target=np.deg2rad(ctrl_cfg.target_angle_deg),
    )
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_pendulum_env(obs_spec=obs_spec, controller=controller)


def seed_env(env: mt.Env) -> None:
    init_cfg = CONFIG.initial_state
    seed_pendulum(env, angle_deg=init_cfg.angle_deg, velocity_deg=init_cfg.velocity_deg)


def summarize(result: mt.PassiveRunResult) -> None:
    recorder = result.recorder
    rows = recorder.rows
    if not rows:
        print(f"Viewer closed. Final simulated time: {result.env.data.time:.3f}s")
        return

    columns = resolve_pendulum_columns(result.env.model)
    column_index = recorder.column_index
    stride = max(1, result.settings.simulation.sample_stride)

    time_idx = column_index[columns["time"]]
    angle_idx = column_index[columns["angle"]]
    velocity_idx = column_index[columns["velocity"]]
    ctrl_idx = column_index[columns["ctrl"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for idx in range(0, len(rows), stride):
        row = rows[idx]
        print(
            "t={:5.3f}s angle={:6.2f}deg vel={:6.2f}deg/s torque={:6.3f}Nm tip_z={:6.3f}m".format(
                float(row[time_idx]),
                float(np.rad2deg(row[angle_idx])),
                float(np.rad2deg(row[velocity_idx])),
                float(row[ctrl_idx]),
                float(row[tip_z_idx]),
            )
        )

    times = np.array([row[time_idx] for row in rows], dtype=float)
    tip_z = np.array([row[tip_z_idx] for row in rows], dtype=float)
    print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")
    print(f"Executed {result.steps} steps; final simulated time: {result.env.data.time:.3f}s")


HARNESS = mt.PassiveRunHarness(
    build_env,
    description="Pendulum PD example (MuJoCo Template)",
    seed_fn=seed_env,
    probes=make_tip_probes,
    start_message="Running pendulum rollout...",
)


def main(argv: list[str] | None = None) -> None:
    init_cfg = CONFIG.initial_state
    ctrl_cfg = CONFIG.controller
    print(
        "Initial angle: {:.2f} deg; velocity: {:.2f} deg/s; target: {:.2f} deg".format(
            init_cfg.angle_deg, init_cfg.velocity_deg, ctrl_cfg.target_angle_deg
        )
    )

    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
