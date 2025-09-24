from __future__ import annotations

import sys

import numpy as np

import mujoco_template as mt
from pendulum_common import initialize_state as seed_pendulum, make_env as make_pendulum_env

LOG_COLUMNS = ("time_s", "qpos_rad", "qvel_rad", "ctrl", "tip_z_m")
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


def _extract_sample(result: mt.StepResult) -> tuple[float, float, float, float, float]:
    obs = result.obs
    time_s = float(obs["time"][0])
    angle = float(obs["qpos"][0])
    velocity = float(obs["qvel"][0])
    ctrl = float(obs["ctrl"][0])
    tip_z = float(obs["sites_pos"][0, 2])
    return (time_s, angle, velocity, ctrl, tip_z)


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
    samples: list[tuple[float, float, float, float, float]] = []

    with mt.TrajectoryLogger(options.log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            row = logger.log(result)
            samples.append(row)

        mt.run_passive_headless(env, max_steps=max_steps, hooks=on_step)

    for idx in range(0, len(samples), SAMPLE_STRIDE):
        time_s, angle_rad, vel_rad, torque, tip_z = samples[idx]
        print(
            "t={:5.3f}s angle={:6.2f}deg vel={:6.2f}deg/s torque={:6.3f}Nm tip_z={:6.3f}m".format(
                time_s, np.rad2deg(angle_rad), np.rad2deg(vel_rad), torque, tip_z
            )
        )

    if samples:
        times = np.array([s[0] for s in samples], dtype=float)
        tip_z = np.array([s[4] for s in samples], dtype=float)
        print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


def run_viewer(env: mt.Env, options: mt.PassiveRunCLIOptions) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    with mt.TrajectoryLogger(options.log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            logger.log(result)

        try:
            mt.run_passive_viewer(env, duration=options.duration, hooks=on_step)
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
