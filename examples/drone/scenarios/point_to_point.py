from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import mujoco_template as mt

from ..controllers import DroneLQRController
from ..drone_common import make_env, make_navigation_probes
from ..drone_config import CONFIG


def _require_lqr_controller(controller: mt.Controller) -> DroneLQRController:
    if not isinstance(controller, DroneLQRController):
        raise mt.TemplateError("DroneLQRController is required for this harness.")
    return controller


def build_env(config: SimpleNamespace = CONFIG) -> mt.Env:
    ctrl_cfg = config.controller
    traj_cfg = config.trajectory
    controller = DroneLQRController(
        ctrl_cfg,
        start_position=traj_cfg.start_position_m,
        start_orientation=traj_cfg.start_orientation_wxyz,
        start_velocity=traj_cfg.start_velocity_mps,
        start_angular_velocity=traj_cfg.start_angular_velocity_radps,
    )
    obs_spec = mt.ObservationSpec(include_ctrl=True, include_sensordata=False, include_time=True)
    return make_env(obs_spec=obs_spec, controller=controller)


def seed_env(env: mt.Env, config: SimpleNamespace = CONFIG) -> None:
    del config  # Configuration is baked into the controller during build_env.
    env.reset()
    controller = _require_lqr_controller(env.controller)
    env.data.qpos[:] = controller.qpos_start
    env.data.qvel[:] = controller.qvel_start
    env.data.ctrl[: controller.ctrl_equilibrium.shape[0]] = controller.ctrl_equilibrium
    env.handle.forward()


def summarize(result: mt.PassiveRunResult) -> None:
    controller = _require_lqr_controller(result.env.controller)
    recorder = result.recorder
    rows = recorder.rows
    goal = controller.goal_position

    if rows:
        column_index = recorder.column_index
        distance_idx = column_index.get("goal_distance_m")
        if distance_idx is not None:
            distances = np.array([row[distance_idx] for row in rows], dtype=float)
            print(
                "Goal distance min {:.3f} m | max {:.3f} m | final {:.3f} m".format(
                    float(distances.min()), float(distances.max()), float(distances[-1])
                )
            )

    final_position = np.array(result.env.data.qpos[:3], dtype=float)
    final_orientation = np.array(result.env.data.qpos[3:7], dtype=float)
    final_velocity = np.array(result.env.data.qvel[:3], dtype=float)
    goal_velocity = controller.goal_velocity
    goal_angular_velocity = controller.goal_angular_velocity
    goal_orientation = controller.goal_orientation

    qvel_goal = controller.qvel_goal
    pos_dim = min(3, qvel_goal.size)
    rot_dim = min(3, max(0, qvel_goal.size - pos_dim))
    final_qvel = np.array(result.env.data.qvel, dtype=float)
    final_linear = np.zeros(3)
    final_linear[:pos_dim] = final_qvel[:pos_dim]
    final_angular = np.zeros(3)
    if rot_dim > 0:
        final_angular[:rot_dim] = final_qvel[pos_dim : pos_dim + rot_dim]

    print(
        "Final position [{:.3f}, {:.3f}, {:.3f}] m | target [{:.3f}, {:.3f}, {:.3f}] m".format(
            *final_position, *goal
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
            float(np.linalg.norm(final_velocity)), float(result.env.data.time), result.steps
        )
    )


HARNESS = mt.PassiveRunHarness(
    build_env,
    description="Skydio X2 drone point-to-point flight via LQR (MuJoCo Template)",
    seed_fn=seed_env,
    probes=make_navigation_probes,
    start_message="Running drone LQR rollout...",
)


__all__ = ["HARNESS", "build_env", "seed_env", "summarize"]

