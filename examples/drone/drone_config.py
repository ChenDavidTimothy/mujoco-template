from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

from drone_common import quat_wxyz_from_body_euler

RUN_SETTINGS = mt.PassiveRunSettings(
    simulation=mt.SimulationSettings(max_steps=4000, duration_seconds=8.0, sample_stride=80),
    video=mt.VideoSettings(
        enabled=False,
        path=Path("drone_lqr.mp4"),
        fps=60.0,
        width=1280,
        height=720,
        crf=20,
        preset="medium",
        tune=None,
        faststart=True,
        capture_initial_frame=True,
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,
            zoom_policy="distance",
            azimuth=120.0,
            elevation=-15.0,
            distance=4.0,
            lookat=(0.0, 0.0, 1.0),
            min_distance=4.0,
            max_distance=14.0,
            safety_margin=0.2,
            widen_threshold=0.82,
            tighten_threshold=0.62,
            smoothing_time_constant=0.05,
            recenter_axis="x",
            recenter_time_constant=1.0,
            points_of_interest=(
                "body:x2",
            ),
        ),
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("drone_lqr.csv"), store_rows=True),
)

TRAJECTORY = SimpleNamespace(
    start_position_m=(0.0, 0.0, 0.3),
    start_orientation_wxyz=quat_wxyz_from_body_euler(yaw_deg=0.0),
    start_velocity_mps=(0.0, 0.0, 0.0),
    start_angular_velocity_radps=(0.0, 0.0, 0.0),
    goal_position_m=(0.0, 0.0, 0.3),
    goal_orientation_wxyz=quat_wxyz_from_body_euler(yaw_deg=90.0),
    goal_velocity_mps=(0.0, 0.0, 0.0),
    goal_angular_velocity_radps=(0.0, 0.0, 0.0),
)

CONTROLLER = SimpleNamespace(
    keyframe="hover",
    linearization_eps=1e-4,
    position_weight=(10.0, 10.0, 10.0),
    orientation_weight=(10.0, 10.0, 20.0),
    velocity_weight=(8.0, 8.0, 6.0),
    angular_velocity_weight=(6.0, 6.0, 10.0),
    control_weight=2.0,
    yaw_control_scale=4.0,
    yaw_integral_gain=3.0,
    yaw_integral_limit=3.0,
    clip_controls=True,
    goal_position_m=TRAJECTORY.goal_position_m,
    goal_orientation_wxyz=TRAJECTORY.goal_orientation_wxyz,
    goal_velocity_mps=TRAJECTORY.goal_velocity_mps,
    goal_angular_velocity_radps=TRAJECTORY.goal_angular_velocity_radps,
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, trajectory=TRAJECTORY, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "TRAJECTORY", "CONTROLLER"]
