from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import mujoco_template as mt
from pendulum_common import (
    initialize_state as seed_pendulum,
    make_env as make_pendulum_env,
    make_tip_probes,
    resolve_pendulum_columns,
)
from pendulum_passive_config import (
    CONFIG,
    LoggingConfig,
    PendulumPassiveConfig,
    SimulationConfig,
    VideoConfig,
    ViewerConfig,
)


def _video_settings_from_config(
    video_cfg: VideoConfig,
    force_enable: bool,
) -> mt.VideoEncoderSettings | None:
    if not (video_cfg.enabled or force_enable):
        return None
    return mt.VideoEncoderSettings(
        path=video_cfg.path,
        fps=video_cfg.fps,
        width=video_cfg.width,
        height=video_cfg.height,
        crf=video_cfg.crf,
        preset=video_cfg.preset,
        tune=video_cfg.tune,
        faststart=video_cfg.faststart,
        capture_initial_frame=video_cfg.capture_initial_frame,
    )


def _log_path_from_config(logging_cfg: LoggingConfig, force_enable: bool) -> Path | None:
    if not (logging_cfg.enabled or force_enable):
        return None
    return logging_cfg.path


def _viewer_requested(viewer_cfg: ViewerConfig, force_enable: bool) -> bool:
    return bool(viewer_cfg.enabled or force_enable)


def build_env(config: PendulumPassiveConfig) -> mt.Env:
    obs_spec = mt.ObservationSpec(
        include_ctrl=False,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_pendulum_env(obs_spec=obs_spec)


def run_headless(
    env: mt.Env,
    sim_cfg: SimulationConfig,
    video_settings: mt.VideoEncoderSettings | None,
    log_path: Path | None,
) -> None:
    max_steps = sim_cfg.headless_max_steps
    if sim_cfg.headless_duration_seconds is not None:
        timestep = float(env.model.opt.timestep)
        max_steps = max(1, int(round(sim_cfg.headless_duration_seconds / timestep)))

    print("Running passive pendulum rollout (headless)...")
    probes = make_tip_probes(env)
    columns = resolve_pendulum_columns(env.model)

    with mt.StateControlRecorder(env, log_path=log_path, probes=probes) as recorder:
        hooks = [recorder]
        if video_settings is not None:
            exporter = mt.VideoExporter(env, video_settings)
            steps = mt.run_passive_video(env, exporter, max_steps=max_steps, hooks=hooks)
            print(f"Exported {steps} steps to {video_settings.path}")
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

    for idx in range(0, len(rows), sim_cfg.sample_stride):
        row = rows[idx]
        print(
            f"t={row[time_idx]:5.3f}s angle={np.rad2deg(row[angle_idx]):6.2f}deg "
            f"vel={np.rad2deg(row[velocity_idx]):6.2f}deg/s tip_z={row[tip_z_idx]:6.3f}m"
        )

    times = np.array([row[time_idx] for row in rows], dtype=float)
    tip_z = np.array([row[tip_z_idx] for row in rows], dtype=float)
    print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")


def run_viewer(env: mt.Env, viewer_cfg: ViewerConfig, log_path: Path | None) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    probes = make_tip_probes(env)
    with mt.StateControlRecorder(env, log_path=log_path, store_rows=False, probes=probes) as recorder:
        try:
            mt.run_passive_viewer(env, duration=viewer_cfg.duration_seconds, hooks=recorder)
        except mt.TemplateError as exc:  # pragma: no cover - viewer availability depends on platform
            raise SystemExit(str(exc)) from exc

    print("Viewer closed. Final simulated time: {:.3f}s".format(env.data.time))


def main(argv: list[str] | None = None) -> None:
    options = mt.parse_passive_run_cli(
        "Passive pendulum example (MuJoCo Template)", args=argv
    )
    config = CONFIG

    sim_cfg = config.simulation
    viewer_cfg = config.viewer
    if options.duration is not None:
        sim_cfg = replace(sim_cfg, headless_duration_seconds=options.duration)
        viewer_cfg = replace(viewer_cfg, duration_seconds=options.duration)

    env = build_env(config)
    init_cfg = config.initial_state
    seed_pendulum(env, angle_deg=init_cfg.angle_deg, velocity_deg=init_cfg.velocity_deg)

    print(
        "Initial pendulum angle: {:.2f} deg; velocity: {:.2f} deg/s".format(
            init_cfg.angle_deg,
            init_cfg.velocity_deg,
        )
    )

    log_path = _log_path_from_config(config.logging, options.logs)
    video_settings = _video_settings_from_config(config.video, options.video)

    if _viewer_requested(viewer_cfg, options.viewer):
        run_viewer(env, viewer_cfg, log_path)
    else:
        run_headless(env, sim_cfg, video_settings, log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


