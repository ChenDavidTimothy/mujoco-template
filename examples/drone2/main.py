"""Straightforward LQR-controlled Skydio X2 flight using mujoco_template."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import mujoco_template as mt


@dataclass(frozen=True)
class DroneRunConfig:
    """Numerical parameters that define the point-to-point flight."""

    keyframe: str = "hover"
    linearization_eps: float = 1e-4
    max_steps: int = 4000
    duration_seconds: float = 8.0
    position_weight: tuple[float, float, float] = (10.0, 10.0, 10.0)
    orientation_weight: tuple[float, float, float] = (10.0, 10.0, 20.0)
    velocity_weight: tuple[float, float, float] = (8.0, 8.0, 6.0)
    angular_velocity_weight: tuple[float, float, float] = (6.0, 6.0, 10.0)
    control_weight: float = 2.0
    yaw_control_scale: float = 6.0
    yaw_proportional_gain: float = 18.0
    yaw_derivative_gain: float = 4.5
    yaw_integral_gain: float = 4.0
    yaw_integral_limit: float = 6.0
    clip_controls: bool = True
    start_position_m: tuple[float, float, float] = (0.0, 0.0, 0.7)
    start_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    start_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_angular_velocity_radps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_position_m: tuple[float, float, float] = (5.0, -4.0, 2.3)
    goal_orientation_wxyz: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    goal_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_angular_velocity_radps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    log_path: Path = Path("drone_lqr.csv")


def _quat_from_body_yaw(yaw_deg: float) -> tuple[float, float, float, float]:
    rot = Rotation.from_euler("xyz", [0.0, 0.0, yaw_deg], degrees=True)
    x, y, z, w = rot.as_quat()
    return float(w), float(x), float(y), float(z)


def _broadcast(scale: tuple[float, ...] | float, size: int) -> np.ndarray:
    arr = np.asarray(scale, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, size)
    if arr.size != size:
        raise ValueError(f"Expected {size} entries, received {arr.size}.")
    if np.any(arr <= 0.0):
        raise ValueError("Feedback weights and scales must be positive.")
    return arr


def _solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    last_error: Exception | None = None
    for scale in (1.0, 10.0, 100.0, 1000.0):
        try:
            P = scipy.linalg.solve_discrete_are(A, B, scale * Q, R)
            return scipy.linalg.solve(R + B.T @ P @ B, B.T @ P @ A, assume_a="sym")
        except Exception as exc:  # pragma: no cover - numerical fallback
            last_error = exc
    raise mt.TemplateError("Failed to compute LQR gains.") from last_error


def _compose_qvel(linear: np.ndarray, angular: np.ndarray) -> np.ndarray:
    return np.concatenate([linear, angular])


def _site_id(model: mt.mj.MjModel, name: str) -> int:
    site_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_SITE, name))
    if site_id < 0:
        raise mt.NameLookupError(f"Site not found: {name}")
    return site_id


def main() -> None:
    cfg = DroneRunConfig(
        start_orientation_wxyz=_quat_from_body_yaw(0.0),
        goal_orientation_wxyz=_quat_from_body_yaw(180.0),
    )

    print(
        "Preparing drone LQR controller (keyframe {} | target [{:.2f}, {:.2f}, {:.2f}] m)".format(
            cfg.keyframe, *np.asarray(cfg.goal_position_m, dtype=float)
        )
    )
    print(
        "Start position [{:.2f}, {:.2f}, {:.2f}] m | duration {} steps".format(
            *np.asarray(cfg.start_position_m, dtype=float), cfg.max_steps
        )
    )
    print(
        "Start orientation [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz".format(
            *np.asarray(cfg.start_orientation_wxyz, dtype=float)
        )
    )
    print(
        "Start velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(cfg.start_velocity_mps, dtype=float),
            *np.asarray(cfg.start_angular_velocity_radps, dtype=float),
        )
    )
    print(
        "Goal velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(cfg.goal_velocity_mps, dtype=float),
            *np.asarray(cfg.goal_angular_velocity_radps, dtype=float),
        )
    )

    scene = Path(__file__).with_name("scene.xml")
    env = mt.Env.from_xml_path(
        str(scene),
        obs_spec=mt.ObservationSpec(include_ctrl=True, include_time=True),
        controller=None,
    )
    env.reset()

    model = env.model
    data = env.data
    nu = int(model.nu)
    nv = int(model.nv)
    pos_dim = min(3, nv)
    rot_dim = min(3, max(0, nv - pos_dim))

    work = mt.mj.MjData(model)
    keyframe_id = int(mt.mj.mj_name2id(model, mt.mj.mjtObj.mjOBJ_KEY, cfg.keyframe))
    if keyframe_id < 0:
        raise mt.ConfigError(f"Keyframe '{cfg.keyframe}' not found in model.")
    mt.mj.mj_resetDataKeyframe(model, work, keyframe_id)
    mt.mj.mj_forward(model, work)

    qpos_hover = np.array(work.qpos, dtype=float)
    ctrl_hover = np.array(work.ctrl[:nu], dtype=float)

    goal_position = np.asarray(cfg.goal_position_m, dtype=float)
    goal_orientation = np.asarray(cfg.goal_orientation_wxyz, dtype=float)
    goal_velocity = np.asarray(cfg.goal_velocity_mps, dtype=float)
    goal_angular_velocity = np.asarray(cfg.goal_angular_velocity_radps, dtype=float)

    qpos_goal = np.array(qpos_hover, copy=True)
    qpos_goal[:3] = goal_position
    qpos_goal[3:7] = goal_orientation
    qvel_goal = _compose_qvel(goal_velocity, goal_angular_velocity)

    start_position = np.asarray(cfg.start_position_m, dtype=float)
    start_orientation = np.asarray(cfg.start_orientation_wxyz, dtype=float)
    start_velocity = np.asarray(cfg.start_velocity_mps, dtype=float)
    start_angular_velocity = np.asarray(cfg.start_angular_velocity_radps, dtype=float)

    qpos_start = np.array(qpos_goal, copy=True)
    qpos_start[:3] = start_position
    qpos_start[3:7] = start_orientation
    qvel_start = _compose_qvel(start_velocity, start_angular_velocity)

    work.qpos[:] = qpos_goal
    work.qvel[:] = qvel_goal
    work.ctrl[:nu] = ctrl_hover
    mt.mj.mj_forward(model, work)

    if cfg.linearization_eps <= 0.0:
        raise mt.ConfigError("linearization_eps must be positive.")
    A, B = mt.linearize_discrete(model, work, eps=cfg.linearization_eps)

    Q = np.diag(
        np.concatenate(
            [
                _broadcast(cfg.position_weight, pos_dim),
                _broadcast(cfg.orientation_weight, rot_dim),
                _broadcast(cfg.velocity_weight, pos_dim),
                _broadcast(cfg.angular_velocity_weight, rot_dim),
            ]
        )
    )
    if cfg.control_weight < 0.0:
        raise mt.ConfigError("control_weight must be non-negative.")
    R = float(cfg.control_weight) * np.eye(nu)

    K = _solve_lqr(A, B, Q, R)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
    print(f"Closed-loop spectral radius: {spectral_radius:.9f}")

    dq_scratch = np.zeros(nv)
    dx_scratch = np.zeros(2 * nv)
    ctrl_delta = np.zeros(nu)
    ctrl_command = np.zeros(nu)

    position_feedback_scale = _broadcast((1.0,), pos_dim)
    orientation_feedback_scale = _broadcast((1.0,), rot_dim) if rot_dim > 0 else np.ones(0)
    velocity_feedback_scale = _broadcast((1.0,), pos_dim)
    angular_velocity_feedback_scale = _broadcast((1.0,), rot_dim) if rot_dim > 0 else np.ones(0)

    yaw_direction = None
    yaw_direction_norm = 0.0
    if rot_dim > 0:
        yaw_idx = nv + pos_dim + rot_dim - 1
        candidate = np.array(B[yaw_idx, :nu], dtype=float)
        norm = float(np.dot(candidate, candidate))
        if norm > 0.0:
            yaw_direction = candidate
            yaw_direction_norm = norm
    yaw_integral = 0.0
    yaw_limit = float(cfg.yaw_integral_limit) if cfg.yaw_integral_limit > 0 else np.inf

    ctrl_low = None
    ctrl_high = None
    if cfg.clip_controls and hasattr(model, "actuator_ctrllimited") and hasattr(model, "actuator_ctrlrange"):
        limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
        ranges = np.asarray(model.actuator_ctrlrange, dtype=float)
        if ranges.shape == (nu, 2):
            ctrl_low = np.where(limited, ranges[:, 0], -np.inf)
            ctrl_high = np.where(limited, ranges[:, 1], np.inf)

    data.qpos[:] = qpos_start
    data.qvel[:] = qvel_start
    data.ctrl[:nu] = ctrl_hover
    mt.mj.mj_forward(model, data)

    imu_site = _site_id(model, "imu")

    probes = (
        mt.DataProbe("imu_x_m", lambda env, _res: float(env.data.site_xpos[imu_site, 0])),
        mt.DataProbe("imu_y_m", lambda env, _res: float(env.data.site_xpos[imu_site, 1])),
        mt.DataProbe("imu_z_m", lambda env, _res: float(env.data.site_xpos[imu_site, 2])),
        mt.DataProbe(
            "goal_distance_m",
            lambda env, _res: float(
                np.linalg.norm(env.data.site_xpos[imu_site] - goal_position)
            ),
        ),
    )

    recorder = mt.StateControlRecorder(env, log_path=cfg.log_path, store_rows=True, probes=probes)

    goal_distances: list[float] = []
    steps = 0

    with recorder:
        while steps < cfg.max_steps:
            mt.mj.mj_differentiatePos(model, dq_scratch, 1.0, qpos_goal, data.qpos)
            dx_scratch[:nv] = dq_scratch
            dx_scratch[nv:] = data.qvel - qvel_goal

            if pos_dim > 0:
                dx_scratch[:pos_dim] *= position_feedback_scale
                dx_scratch[nv : nv + pos_dim] *= velocity_feedback_scale
            if rot_dim > 0:
                start = pos_dim
                stop = pos_dim + rot_dim
                dx_scratch[start:stop] *= orientation_feedback_scale
                vel_start = nv + pos_dim
                vel_stop = vel_start + rot_dim
                dx_scratch[vel_start:vel_stop] *= angular_velocity_feedback_scale

            ctrl_delta[:] = K @ dx_scratch

            if yaw_direction is not None and yaw_direction_norm > 0.0 and rot_dim > 0:
                yaw_error = float(dx_scratch[pos_dim + rot_dim - 1])
                yaw_rate_error = float(dx_scratch[nv + pos_dim + rot_dim - 1])
                if cfg.yaw_integral_gain > 0.0:
                    yaw_integral += yaw_error * float(model.opt.timestep)
                    if not np.isinf(yaw_limit):
                        yaw_integral = float(np.clip(yaw_integral, -yaw_limit, yaw_limit))
                    ctrl_delta += cfg.yaw_integral_gain * yaw_integral * yaw_direction
                if cfg.yaw_proportional_gain > 0.0 or cfg.yaw_derivative_gain > 0.0:
                    yaw_feedback = cfg.yaw_proportional_gain * yaw_error + cfg.yaw_derivative_gain * yaw_rate_error
                    ctrl_delta += yaw_feedback * yaw_direction
                if cfg.yaw_control_scale != 1.0:
                    yaw_component = float(np.dot(ctrl_delta, yaw_direction)) / yaw_direction_norm
                    ctrl_delta += (cfg.yaw_control_scale - 1.0) * yaw_component * yaw_direction

            ctrl_command[:] = ctrl_hover - ctrl_delta
            if ctrl_low is not None and ctrl_high is not None:
                np.clip(ctrl_command, ctrl_low, ctrl_high, out=ctrl_command)
            data.ctrl[:nu] = ctrl_command

            result = env.step(return_obs=True)
            recorder(result)
            goal_distances.append(float(np.linalg.norm(data.site_xpos[imu_site] - goal_position)))
            steps += 1

            if cfg.duration_seconds is not None and data.time >= cfg.duration_seconds:
                break

    if cfg.log_path:
        print(f"Logged trajectory to {cfg.log_path}")

    if goal_distances:
        distances = np.asarray(goal_distances, dtype=float)
        print(
            "Goal distance min {:.3f} m | max {:.3f} m | final {:.3f} m".format(
                float(distances.min()), float(distances.max()), float(distances[-1])
            )
        )

    final_position = np.array(data.qpos[:3], dtype=float)
    final_orientation = np.array(data.qpos[3:7], dtype=float)
    final_velocity = np.array(data.qvel[:3], dtype=float)

    final_linear = np.zeros(3)
    final_angular = np.zeros(3)
    pos_dim_goal = min(3, qvel_goal.size)
    rot_dim_goal = min(3, max(0, qvel_goal.size - pos_dim_goal))
    final_linear[:pos_dim_goal] = np.array(data.qvel[:pos_dim_goal], dtype=float)
    if rot_dim_goal > 0:
        final_angular[:rot_dim_goal] = np.array(
            data.qvel[pos_dim_goal : pos_dim_goal + rot_dim_goal], dtype=float
        )

    print(
        "Final position [{:.3f}, {:.3f}, {:.3f}] m | target [{:.3f}, {:.3f}, {:.3f}] m".format(
            *final_position,
            *goal_position,
        )
    )
    print(
        "Final orientation [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz | target [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz".format(
            *final_orientation,
            *goal_orientation,
        )
    )
    print(
        "Final velocity [{:.3f}, {:.3f}, {:.3f}] m/s | target [{:.3f}, {:.3f}, {:.3f}] m/s".format(
            *final_linear,
            *goal_velocity,
        )
    )
    print(
        "Final angular velocity [{:.3f}, {:.3f}, {:.3f}] rad/s | target [{:.3f}, {:.3f}, {:.3f}] rad/s".format(
            *final_angular,
            *goal_angular_velocity,
        )
    )
    print(
        "Translational speed {:.3f} m/s | simulated {:.3f} s over {} steps".format(
            float(np.linalg.norm(final_velocity)),
            float(data.time),
            steps,
        )
    )


if __name__ == "__main__":
    main()
