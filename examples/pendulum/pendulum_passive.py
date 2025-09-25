from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import mujoco_template as mt
from pendulum_common import (
    initialize_state as seed_pendulum,
    make_env as make_pendulum_env,
    make_tip_probes,
    resolve_pendulum_columns,
)

DEFAULT_HEADLESS_STEPS = 600
SAMPLE_STRIDE = 120
INITIAL_ANGLE_DEG = 90.0
INITIAL_VELOCITY_DEG = 0.0

EXPORT_VIDEO = True
VIDEO_PATH = Path("pendulum_passive.mp4")
VIDEO_FPS = 60.0
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_CRF = 18
VIDEO_PRESET = "medium"
VIDEO_TUNE: str | None = None
VIDEO_FASTSTART = True
VIDEO_CAPTURE_INITIAL_FRAME = True

LOG_PATH: Path | None = None
HEADLESS_DURATION_SECONDS: float | None = None
HEADLESS_MAX_STEPS = DEFAULT_HEADLESS_STEPS
USE_VIEWER = False
VIEWER_DURATION_SECONDS: float | None = None

VIDEO_SETTINGS = (
    mt.VideoEncoderSettings(
        path=VIDEO_PATH,
        fps=VIDEO_FPS,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        crf=VIDEO_CRF,
        preset=VIDEO_PRESET,
        tune=VIDEO_TUNE,
        faststart=VIDEO_FASTSTART,
        capture_initial_frame=VIDEO_CAPTURE_INITIAL_FRAME,
    )
    if EXPORT_VIDEO
    else None
)


def build_env() -> mt.Env:
    obs_spec = mt.ObservationSpec(
        include_ctrl=False,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_pendulum_env(obs_spec=obs_spec)


def run_headless(env: mt.Env) -> None:
    max_steps = HEADLESS_MAX_STEPS
    if HEADLESS_DURATION_SECONDS is not None:
        timestep = float(env.model.opt.timestep)
        max_steps = max(1, int(round(HEADLESS_DURATION_SECONDS / timestep)))

    print("Running passive pendulum rollout (headless)...")
    probes = make_tip_probes(env)
    columns = resolve_pendulum_columns(env.model)

    with mt.StateControlRecorder(env, log_path=LOG_PATH, probes=probes) as recorder:
        hooks = [recorder]
        if VIDEO_SETTINGS is not None:
            exporter = mt.VideoExporter(env, VIDEO_SETTINGS)
            steps = mt.run_passive_video(env, exporter, max_steps=max_steps, hooks=hooks)
            print(f"Exported {steps} steps to {VIDEO_SETTINGS.path}")
        else:
            mt.run_passive_headless(env, max_steps=max_steps, hooks=hooks)
        rows = list(recorder.rows)
        column_index = recorder.column_index

    if not rows:
        print("No simulation steps executed.")
        return

    time_idx = column_index[columns["time"]]
    angle_idx = column_index[columns["angle"]]
    velocity_idx = column_index[columns["velocity"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for idx in range(0, len(rows), SAMPLE_STRIDE):
        row = rows[idx]
        print(
            f"t={row[time_idx]:5.3f}s angle={np.rad2deg(row[angle_idx]):6.2f}deg "
            f"vel={np.rad2deg(row[velocity_idx]):6.2f}deg/s tip_z={row[tip_z_idx]:6.3f}m"
        )

    times = np.array([row[time_idx] for row in rows], dtype=float)
    tip_z = np.array([row[tip_z_idx] for row in rows], dtype=float)
    print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


def run_viewer(env: mt.Env) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    probes = make_tip_probes(env)
    with mt.StateControlRecorder(env, log_path=LOG_PATH, store_rows=False, probes=probes) as recorder:
        try:
            mt.run_passive_viewer(env, duration=VIEWER_DURATION_SECONDS, hooks=recorder)
        except mt.TemplateError as exc:  # pragma: no cover - viewer availability depends on platform
            raise SystemExit(str(exc)) from exc

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main() -> None:
    env = build_env()
    seed_pendulum(env, angle_deg=INITIAL_ANGLE_DEG, velocity_deg=INITIAL_VELOCITY_DEG)

    print(
        "Initial pendulum angle: {:.2f} deg; velocity: {:.2f} deg/s".format(
            INITIAL_ANGLE_DEG, INITIAL_VELOCITY_DEG
        )
    )

    if USE_VIEWER:
        run_viewer(env)
    else:
        run_headless(env)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


