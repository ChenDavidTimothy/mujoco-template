from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import mujoco_template as mt

LOG_COLUMNS = ("time_s", "qpos_rad", "qvel_rad", "ctrl", "tip_z_m")


class PendulumPDController:
    """Simple PD torque controller that stabilizes the pendulum upright."""

    def __init__(self, kp: float = 20.0, kd: float = 4.0, target: float = 0.0) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PD-controlled pendulum demo")
    parser.add_argument("--steps", type=int, default=400, help="Number of headless simulation steps to run.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=80,
        help="Step stride for console logging in headless mode.",
    )
    parser.add_argument("--viewer", action="store_true", help="Launch the interactive viewer instead of running headless.")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional viewer duration limit in seconds (<=0 runs until closed).")
    parser.add_argument("--log-path", type=Path, default=None, help="Optional CSV path for writing trajectory samples.")
    parser.add_argument("--initial-angle-deg", type=float, default=60.0, help="Initial pendulum angle in degrees.")
    parser.add_argument("--initial-velocity-deg", type=float, default=0.0, help="Initial angular velocity in degrees per second.")
    parser.add_argument("--target-deg", type=float, default=0.0, help="PD target angle in degrees.")
    parser.add_argument("--kp", type=float, default=20.0, help="PD proportional gain.")
    parser.add_argument("--kd", type=float, default=4.0, help="PD derivative gain.")
    return parser.parse_args()


def build_env(kp: float, kd: float, target_deg: float) -> mt.Env:
    xml_path = Path(__file__).with_name("pendulum.xml")
    handle = mt.ModelHandle.from_xml_path(str(xml_path))
    controller = PendulumPDController(kp=kp, kd=kd, target=np.deg2rad(target_deg))
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return mt.Env(handle, obs_spec=obs_spec, controller=controller)


def prepare_initial_state(
    env: mt.Env,
    *,
    angle_deg: float,
    velocity_deg: float,
) -> None:
    env.reset()
    env.data.qpos[0] = np.deg2rad(angle_deg)
    env.data.qvel[0] = np.deg2rad(velocity_deg)
    env.handle.forward()


def run_headless(env: mt.Env, steps: int, sample_stride: int, log_path: Path | None) -> None:
    if steps < 1:
        raise mt.ConfigError("steps must be >= 1")
    sample_stride = max(1, sample_stride)

    print("Running pendulum rollout (headless)...")
    samples: list[tuple[float, float, float, float, float]] = []

    with mt.TrajectoryLogger(log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            row = logger.log(result)
            samples.append(row)

        mt.run_passive_headless(env, max_steps=steps, hooks=on_step)

    for idx in range(0, len(samples), sample_stride):
        t, angle, vel, torque, tip_z = samples[idx]
        print(
            "t={:5.3f}s angle={:6.2f}deg vel={:6.2f}deg/s torque={:6.3f}Nm tip_z={:6.3f}m".format(
                t, np.rad2deg(angle), np.rad2deg(vel), torque, tip_z
            )
        )

    if samples:
        times = np.array([s[0] for s in samples], dtype=float)
        tip_z = np.array([s[4] for s in samples], dtype=float)
        print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


def run_viewer(env: mt.Env, duration: float, log_path: Path | None) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    with mt.TrajectoryLogger(log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            logger.log(result)

        try:
            mt.run_passive_viewer(env, duration=duration if duration > 0 else None, hooks=on_step)
        except mt.TemplateError as exc:  # pragma: no cover - viewer availability depends on platform
            raise SystemExit(str(exc)) from exc

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main() -> None:
    args = parse_args()
    env = build_env(args.kp, args.kd, args.target_deg)
    prepare_initial_state(env, angle_deg=args.initial_angle_deg, velocity_deg=args.initial_velocity_deg)

    print(
        "Initial angle: {:.2f} deg; velocity: {:.2f} deg/s; target: {:.2f} deg".format(
            args.initial_angle_deg, args.initial_velocity_deg, args.target_deg
        )
    )

    if args.viewer:
        run_viewer(env, args.duration, args.log_path)
    else:
        run_headless(env, args.steps, args.sample_stride, args.log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
