from __future__ import annotations

import argparse
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
from pendulum_config import (
    CONFIG,
    LoggingConfig,
    PendulumConfig,
    SimulationConfig,
    VideoConfig,
    ViewerConfig,
)


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pendulum PD example (MuJoCo Template)")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Force enable the interactive viewer using config defaults.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Force enable video export using config defaults.",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Force enable CSV logging using config defaults.",
    )
    return parser.parse_args(argv)


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


def build_env(config: PendulumConfig) -> mt.Env:
    ctrl_cfg = config.controller
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

    print("Running pendulum rollout (headless)...")
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
    ctrl_idx = column_index[columns["ctrl"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for idx in range(0, len(rows), sim_cfg.sample_stride):
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
    cli = parse_cli(argv)
    config = CONFIG

    env = build_env(config)
    init_cfg = config.initial_state
    seed_pendulum(env, angle_deg=init_cfg.angle_deg, velocity_deg=init_cfg.velocity_deg)

    ctrl_cfg = config.controller
    print(
        "Initial angle: {:.2f} deg; velocity: {:.2f} deg/s; target: {:.2f} deg".format(
            init_cfg.angle_deg, init_cfg.velocity_deg, ctrl_cfg.target_angle_deg
        )
    )

    log_path = _log_path_from_config(config.logging, cli.logs)
    video_settings = _video_settings_from_config(config.video, cli.video)
    viewer_cfg = config.viewer

    if _viewer_requested(viewer_cfg, cli.viewer):
        run_viewer(env, viewer_cfg, log_path)
    else:
        run_headless(env, config.simulation, video_settings, log_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
