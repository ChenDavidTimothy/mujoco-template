from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import mujoco_template as mt

LOG_COLUMNS = ("time_s", "qpos_rad", "qvel_rad", "tip_z_m")


def _extract_sample(result: mt.StepResult) -> tuple[float, float, float, float]:
    obs = result.obs
    time_s = float(obs["time"][0])
    angle = float(obs["qpos"][0])
    velocity = float(obs["qvel"][0])
    tip_z = float(obs["sites_pos"][0, 2])
    return (time_s, angle, velocity, tip_z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Passive pendulum simulation example")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "Simulation duration in seconds; use <=0 to run until the viewer closes "
            "(viewer mode) or for a default 600 steps headless."
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch the interactive MuJoCo viewer instead of logging samples.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=120,
        help="Step stride for console logging in headless mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--initial-angle-deg",
        type=float,
        default=90.0,
        help="Initial pendulum angle in degrees; 0 points down.",
    )
    parser.add_argument(
        "--initial-velocity-deg",
        type=float,
        default=0.0,
        help="Initial angular velocity in degrees per second.",
    )
    parser.add_argument(
        "--initial-angle-noise-deg",
        type=float,
        default=0.0,
        help="Gaussian noise (std dev in degrees) added to the initial angle.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional CSV path for writing trajectory samples.",
    )
    return parser.parse_args()


def prepare_initial_state(
    env: mt.Env,
    *,
    base_angle_deg: float,
    base_velocity_deg: float,
    rng: np.random.Generator | None = None,
    angle_noise_deg: float = 0.0,
) -> float:
    env.reset()
    angle_deg = base_angle_deg
    if angle_noise_deg > 0.0 and rng is not None:
        angle_deg += float(rng.normal(0.0, angle_noise_deg))
    env.data.qpos[0] = np.deg2rad(angle_deg)
    env.data.qvel[0] = np.deg2rad(base_velocity_deg)
    env.handle.forward()
    return angle_deg


def build_env() -> mt.Env:
    xml_path = Path(__file__).with_name("pendulum.xml")
    handle = mt.ModelHandle.from_xml_path(str(xml_path))
    obs_spec = mt.ObservationSpec(
        include_ctrl=False,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return mt.Env(handle, obs_spec=obs_spec)


def run_headless(
    env: mt.Env,
    duration: float,
    sample_stride: int,
    log_path: Path | None,
) -> None:
    timestep = float(env.model.opt.timestep)
    if duration <= 0:
        max_steps = 600
    else:
        max_steps = max(1, int(round(duration / timestep)))
    sample_stride = max(1, sample_stride)

    print("Running passive pendulum rollout (headless)...")
    samples: list[tuple[float, float, float, float]] = []

    with mt.TrajectoryLogger(log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            row = logger.log(result)
            samples.append(row)

        try:
            mt.run_passive_headless(env, max_steps=max_steps, hooks=on_step)
        except Exception as exc:  # pragma: no cover - propagate message upwards
            print(f"Simulation step failed: {exc}", file=sys.stderr)
            raise

    for idx in range(0, len(samples), sample_stride):
        time_s, angle_rad, vel_rad, tip_z = samples[idx]
        print(
            f"t={time_s:5.3f}s angle={np.rad2deg(angle_rad):6.2f}deg "
            f"vel={np.rad2deg(vel_rad):6.2f}deg/s tip_z={tip_z:6.3f}m"
        )

    if samples:
        times = np.array([s[0] for s in samples], dtype=float)
        tip_z = np.array([s[3] for s in samples], dtype=float)
        print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


def run_viewer(
    env: mt.Env,
    duration: float,
    log_path: Path | None,
) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    with mt.TrajectoryLogger(log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            logger.log(result)

        try:
            mt.run_passive_viewer(
                env,
                duration=duration if duration > 0 else None,
                hooks=on_step,
            )
        except mt.TemplateError as exc:  # pragma: no cover - viewer availability depends on platform
            raise SystemExit(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - propagate message upwards
            print(f"Simulation step failed: {exc}", file=sys.stderr)
            raise

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = build_env()
    angle_deg = prepare_initial_state(
        env,
        base_angle_deg=args.initial_angle_deg,
        base_velocity_deg=args.initial_velocity_deg,
        rng=rng,
        angle_noise_deg=args.initial_angle_noise_deg,
    )
    print(f"Initial pendulum angle: {angle_deg:.2f} deg; velocity: {args.initial_velocity_deg:.2f} deg/s")

    if args.viewer:
        run_viewer(env, args.duration, args.log_path)
    else:
        run_headless(env, args.duration, args.sample_stride, args.log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
