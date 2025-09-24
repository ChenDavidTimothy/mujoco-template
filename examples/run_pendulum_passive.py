from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mujoco_template as mt

LOG_COLUMNS = ("time_s", "qpos_rad", "qvel_rad", "tip_z_m")
DEFAULT_HEADLESS_STEPS = 600
SAMPLE_STRIDE = 120
INITIAL_ANGLE_DEG = 90.0
INITIAL_VELOCITY_DEG = 0.0


def _extract_sample(result: mt.StepResult) -> tuple[float, float, float, float]:
    obs = result.obs
    time_s = float(obs["time"][0])
    angle = float(obs["qpos"][0])
    velocity = float(obs["qvel"][0])
    tip_z = float(obs["sites_pos"][0, 2])
    return (time_s, angle, velocity, tip_z)


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


def initialize_state(env: mt.Env) -> None:
    env.reset()
    env.data.qpos[0] = np.deg2rad(INITIAL_ANGLE_DEG)
    env.data.qvel[0] = np.deg2rad(INITIAL_VELOCITY_DEG)
    env.handle.forward()


def run_headless(env: mt.Env, options: mt.PassiveRunCLIOptions) -> None:
    timestep = float(env.model.opt.timestep)
    if options.duration is None:
        max_steps = DEFAULT_HEADLESS_STEPS
    else:
        max_steps = max(1, int(round(options.duration / timestep)))

    print("Running passive pendulum rollout (headless)...")
    samples: list[tuple[float, float, float, float]] = []

    with mt.TrajectoryLogger(options.log_path, LOG_COLUMNS, _extract_sample) as logger:
        def on_step(result: mt.StepResult) -> None:
            row = logger.log(result)
            samples.append(row)

        mt.run_passive_headless(env, max_steps=max_steps, hooks=on_step)

    for idx in range(0, len(samples), SAMPLE_STRIDE):
        time_s, angle_rad, vel_rad, tip_z = samples[idx]
        print(
            f"t={time_s:5.3f}s angle={np.rad2deg(angle_rad):6.2f}deg "
            f"vel={np.rad2deg(vel_rad):6.2f}deg/s tip_z={tip_z:6.3f}m"
        )

    if samples:
        times = np.array([s[0] for s in samples], dtype=float)
        tip_z = np.array([s[3] for s in samples], dtype=float)
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
    options = mt.parse_passive_run_cli("Passive pendulum simulation example")
    env = build_env()
    initialize_state(env)

    print(
        "Initial pendulum angle: {:.2f} deg; velocity: {:.2f} deg/s".format(
            INITIAL_ANGLE_DEG, INITIAL_VELOCITY_DEG
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
