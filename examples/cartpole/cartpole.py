from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

import mujoco_template as mt
from cartpole_common import initialize_state as seed_cartpole, make_env as make_cartpole_env
from cartpole_config import (
    CONFIG,
    CartPoleConfig,
    LoggingConfig,
    SimulationConfig,
    VideoConfig,
    ViewerConfig,
)


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cartpole example (MuJoCo Template)")
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


class CartPolePIDController:
    """PID balance controller applying horizontal force to the cart."""

    def __init__(
        self,
        *,
        angle_kp: float = 16.66,
        angle_kd: float = 4.45,
        angle_ki: float = 0.0,
        position_kp: float = 1.11,
        position_kd: float = 2.20,
        integral_limit: float = 5.0,
    ) -> None:
        self.capabilities = mt.ControllerCapabilities(control_space=mt.ControlSpace.TORQUE)
        self.angle_kp = float(angle_kp)
        self.angle_kd = float(angle_kd)
        self.angle_ki = float(angle_ki)
        self.position_kp = float(position_kp)
        self.position_kd = float(position_kd)
        self.integral_limit = float(abs(integral_limit))
        self._integral_term = 0.0
        self._dt = 0.0

    def prepare(self, model: mt.mj.MjModel, data: mt.mj.MjData) -> None:
        if model.nu != 1:
            raise mt.CompatibilityError("CartPolePIDController expects a single actuator driving the cart.")
        self._integral_term = 0.0
        self._dt = float(model.opt.timestep)

    def __call__(self, model: mt.mj.MjModel, data: mt.mj.MjData, t: float) -> None:
        cart_x = float(data.qpos[0])
        pole_angle = float(data.qpos[1])
        cart_vel = float(data.qvel[0])
        pole_ang_vel = float(data.qvel[1])

        if self.angle_ki != 0.0:
            self._integral_term += pole_angle * self._dt
            self._integral_term = float(np.clip(self._integral_term, -self.integral_limit, self.integral_limit))
        else:
            self._integral_term = 0.0

        # Linear state feedback (gains tuned via LQR) on cart position/velocity and pole angle/velocity.
        force = (
            self.position_kp * cart_x
            + self.position_kd * cart_vel
            + self.angle_kp * pole_angle
            + self.angle_kd * pole_ang_vel
            + self.angle_ki * self._integral_term
        )

        if hasattr(model, "actuator_ctrlrange") and model.actuator_ctrlrange.size >= 2:
            low, high = model.actuator_ctrlrange[0]
            force = float(np.clip(force, low, high))
        data.ctrl[0] = force


def _require_site_id(model: mt.mj.MjModel, name: str) -> int:
    site_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_SITE, name))
    if site_id < 0:
        raise mt.NameLookupError(f"Site not found in model: {name}")
    return site_id


def _resolve_joint_label(model: mt.mj.MjModel, name: str) -> str:
    joint_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_JOINT, name))
    if joint_id < 0:
        raise mt.NameLookupError(f"Joint not found in model: {name}")
    resolved = mt.mj.mj_id2name(model, mt.mj.mjtObj.mjOBJ_JOINT, joint_id)
    return resolved if resolved is not None else f"joint_{joint_id}"


def _resolve_actuator_label(model: mt.mj.MjModel, name: str) -> str:
    actuator_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_ACTUATOR, name))
    if actuator_id < 0:
        raise mt.NameLookupError(f"Actuator not found in model: {name}")
    resolved = mt.mj.mj_id2name(model, mt.mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    return resolved if resolved is not None else f"actuator_{actuator_id}"


def _make_tip_probes(env: mt.Env) -> tuple[mt.DataProbe, ...]:
    tip_id = _require_site_id(env.model, "tip")
    return (
        mt.DataProbe("tip_x_m", lambda e, _r, sid=tip_id: float(e.data.site_xpos[sid, 0])),
        mt.DataProbe("tip_z_m", lambda e, _r, sid=tip_id: float(e.data.site_xpos[sid, 2])),
    )


def _resolve_primary_columns(model: mt.mj.MjModel) -> dict[str, str]:
    slider_label = _resolve_joint_label(model, "slider")
    hinge_label = _resolve_joint_label(model, "hinge")
    actuator_label = _resolve_actuator_label(model, "cart_force")
    return {
        "time": "time_s",
        "cart_pos": f"qpos[{slider_label}]",
        "cart_vel": f"qvel[{slider_label}]",
        "pole_angle": f"qpos[{hinge_label}]",
        "pole_vel": f"qvel[{hinge_label}]",
        "force": f"ctrl[{actuator_label}]",
        "tip_x": "tip_x_m",
        "tip_z": "tip_z_m",
    }


def build_env(config: CartPoleConfig) -> mt.Env:
    controller = CartPolePIDController(**asdict(config.controller))
    obs_spec = mt.ObservationSpec(
        include_ctrl=True,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_cartpole_env(obs_spec=obs_spec, controller=controller)


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

    print("Running cartpole PID rollout (headless)...")
    probes = _make_tip_probes(env)
    columns = _resolve_primary_columns(env.model)

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
    cart_pos_idx = column_index[columns["cart_pos"]]
    cart_vel_idx = column_index[columns["cart_vel"]]
    pole_angle_idx = column_index[columns["pole_angle"]]
    pole_vel_idx = column_index[columns["pole_vel"]]
    force_idx = column_index[columns["force"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for idx in range(0, len(rows), sim_cfg.sample_stride):
        row = rows[idx]
        print(
            "t={:5.2f}s cart={:6.3f}m cartdot={:6.3f}m/s pole={:6.2f}deg poledot={:6.2f}deg/s force={:6.2f}N tip_z={:6.3f}m".format(
                float(row[time_idx]),
                float(row[cart_pos_idx]),
                float(row[cart_vel_idx]),
                float(np.rad2deg(row[pole_angle_idx])),
                float(np.rad2deg(row[pole_vel_idx])),
                float(row[force_idx]),
                float(row[tip_z_idx]),
            )
        )

    pole_angles_deg = np.array([np.rad2deg(row[pole_angle_idx]) for row in rows], dtype=float)
    tip_z = np.array([row[tip_z_idx] for row in rows], dtype=float)
    print(
        "Pole angle range: {:.2f} deg to {:.2f} deg | Tip height range: {:.3f} m to {:.3f} m".format(
            float(pole_angles_deg.min()),
            float(pole_angles_deg.max()),
            float(tip_z.min()),
            float(tip_z.max()),
        )
    )


def run_viewer(env: mt.Env, viewer_cfg: ViewerConfig, log_path: Path | None) -> None:
    print("Launching MuJoCo viewer... close the window to exit.")

    probes = _make_tip_probes(env)
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
    seed_cfg = config.initial_state
    seed_cartpole(
        env,
        cart_position=seed_cfg.cart_position,
        cart_velocity=seed_cfg.cart_velocity,
        pole_angle_rad=np.deg2rad(seed_cfg.pole_angle_deg),
        pole_velocity=np.deg2rad(seed_cfg.pole_velocity_deg),
    )

    print(
        "Initial cart x: {:.3f} m | pole angle: {:.2f} deg | pole velocity: {:.2f} deg/s".format(
            seed_cfg.cart_position,
            seed_cfg.pole_angle_deg,
            seed_cfg.pole_velocity_deg,
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
