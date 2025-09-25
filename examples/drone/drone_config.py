from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings(
    simulation=mt.SimulationSettings(max_steps=4000, duration_seconds=8.0, sample_stride=80),
    video=mt.VideoSettings(
        enabled=True,
        path=Path("drone_lqr.mp4"),
        fps=60.0,
        width=1280,
        height=720,
        crf=20,
        preset="medium",
        tune=None,
        faststart=True,
        capture_initial_frame=True,
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("drone_lqr.csv"), store_rows=True),
)

TRAJECTORY = SimpleNamespace(
    start_position_m=(0.0, 0.0, 0.3),
    start_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    goal_position_m=(1.0, 0.0, 0.6),
    goal_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
)

CONTROLLER = SimpleNamespace(
    keyframe="hover",
    linearization_eps=1e-6,
    position_weight=18.0,
    orientation_weight=4.0,
    velocity_weight=8.0,
    angular_velocity_weight=2.0,
    control_weight=0.4,
    clip_controls=True,
    goal_position_m=TRAJECTORY.goal_position_m,
    goal_orientation_wxyz=TRAJECTORY.goal_orientation_wxyz,
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, trajectory=TRAJECTORY, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "TRAJECTORY", "CONTROLLER"]
