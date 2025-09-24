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

DEFAULT_HEADLESS_STEPS = 400
SAMPLE_STRIDE = 80
INITIAL_ANGLE_DEG = 60.0
INITIAL_VELOCITY_DEG = 0.0
TARGET_ANGLE_DEG = 0.0
KP = 20.0
KD = 4.0


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
    controller = PendulumPDController(kp=KP, kd=KD, target=np.deg2rad(TARGET_ANGLE_DEG))
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_pendulum_env(obs_spec=obs_spec, controller=controller)


def run_headless(env: mt.Env, options: mt.PassiveRunCLIOptions) -> None:
    timestep = float(env.model.opt.timestep)
    if options.duration is None:
        max_steps = DEFAULT_HEADLESS_STEPS
    else:
        max_steps = max(1, int(round(options.duration / timestep)))

    print("Running pendulum rollout (headless)...")
    probes = make_tip_probes(env)
    columns = resolve_pendulum_columns(env.model)

    with mt.StateControlRecorder(env, log_path=options.log_path, probes=probes) as recorder:
        mt.run_passive_headless(env, max_steps=max_steps, hooks=recorder)
        rows = list(recorder.rows)
        column_index = recorder.column_index

    if not rows:
        print("No simulation steps executed.")
        return

    time_idx = column_index[columns["time"]]
    angle_idx = column_index[columns["angle"]]
    velocity_idx = column_index[columns["velocity"]]
    ctrl_idx = column_index[columns["ctrl"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for idx in range(0, len(rows), SAMPLE_STRIDE):
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


def run_viewer(env: mt.Env, options: mt.PassiveRunCLIOptions) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    probes = make_tip_probes(env)
    with mt.StateControlRecorder(env, log_path=options.log_path, store_rows=False, probes=probes) as recorder:
        try:
            mt.run_passive_viewer(env, duration=options.duration, hooks=recorder)
        except mt.TemplateError as exc:  # pragma: no cover - viewer availability depends on platform
            raise SystemExit(str(exc)) from exc

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main() -> None:
    options = mt.parse_passive_run_cli("PD-controlled pendulum demo")
    env = build_env()
    seed_pendulum(env, angle_deg=INITIAL_ANGLE_DEG, velocity_deg=INITIAL_VELOCITY_DEG)

    print(
        "Initial angle: {:.2f} deg; velocity: {:.2f} deg/s; target: {:.2f} deg".format(
            INITIAL_ANGLE_DEG, INITIAL_VELOCITY_DEG, TARGET_ANGLE_DEG
        )
    )

    if options.viewer:
        run_viewer(env, options)
    else:
        run_headless(env, options)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
