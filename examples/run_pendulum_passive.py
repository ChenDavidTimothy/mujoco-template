from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import mujoco_template as mt


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
    return parser.parse_args()


def build_env() -> mt.Env:
    xml_path = Path(__file__).with_name("pendulum.xml")
    handle = mt.ModelHandle.from_xml_path(str(xml_path))
    obs_spec = mt.ObservationSpec(
        include_ctrl=False,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    env = mt.Env(handle, obs_spec=obs_spec)
    env.reset()
    env.data.qpos[0] = np.deg2rad(90.0)
    env.data.qvel[0] = 0.0
    env.handle.forward()
    return env


def run_headless(env: mt.Env, duration: float, sample_stride: int) -> None:
    timestep = float(env.model.opt.timestep)
    if duration <= 0:
        steps = 600
    else:
        steps = max(1, int(round(duration / timestep)))
    sample_stride = max(1, sample_stride)

    print("Running passive pendulum rollout (headless)...")
    samples = []
    for _ in range(steps):
        res = env.step()
        obs = res.obs
        samples.append(
            (
                float(obs["time"][0]),
                float(obs["qpos"][0]),
                float(obs["qvel"][0]),
                float(obs["sites_pos"][0, 2]),
            )
        )

    for idx in range(0, len(samples), sample_stride):
        t, angle, vel, tip_z = samples[idx]
        print(
            f"t={t:5.3f}s angle={np.rad2deg(angle):6.2f}deg vel={np.rad2deg(vel):6.2f}deg/s tip_z={tip_z:6.3f}m"
        )

    times = np.array([s[0] for s in samples], dtype=float)
    tip_z = np.array([s[3] for s in samples], dtype=float)
    print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


def run_viewer(env: mt.Env, duration: float) -> None:
    try:
        import mujoco.viewer as mj_viewer
    except Exception as exc:  # pragma: no cover - viewer availability depends on platform
        raise SystemExit(f"MuJoCo viewer is unavailable: {exc}") from exc

    timestep = float(env.model.opt.timestep)
    print("Launching MuJoCo viewer... close the window to exit.")

    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            env.step()
            viewer.sync()
            if duration > 0 and env.data.time >= duration:
                break
            # Sleep to keep physics roughly in real time; skip if stepping overruns.
            remainder = timestep - (time.perf_counter() - step_start)
            if remainder > 0:
                time.sleep(remainder)

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main() -> None:
    args = parse_args()
    env = build_env()

    if args.viewer:
        run_viewer(env, args.duration)
    else:
        run_headless(env, args.duration, args.sample_stride)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
