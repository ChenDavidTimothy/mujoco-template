from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt


RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    logging=True,
    simulation_overrides=dict(max_steps=2000, duration_seconds=None, sample_stride=50),
    video_overrides=dict(
        path=Path("cartpole.mp4"),
        fps=60.0,
        width=1280,
        height=720,
        crf=18,
        preset="medium",
        tune=None,
        faststart=True,
        capture_initial_frame=True,
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,
            zoom_policy="distance",
            azimuth=90.0,
            elevation=-20.0,
            distance=1.0,
            lookat=(0.0, 0.0, 0),
            min_distance=1.0,
            max_distance=12.0,
            safety_margin=0.1,
            widen_threshold=0.75,
            tighten_threshold=0.55,
            smoothing_time_constant=0.2,
            recenter_axis="x",
            recenter_time_constant=1.5,
            points_of_interest=("body:cart", "site:tip"),
        ),
    ),
    logging_overrides=dict(path=Path("cartpole.csv"), store_rows=True),
)


INITIAL_STATE = SimpleNamespace(
    cart_position=0.0,
    cart_velocity=0.0,
    pole_angle_deg=30.0,
    pole_velocity_deg=0.0,
)


CONTROLLER = SimpleNamespace(
    angle_kp=16.66,
    angle_kd=4.45,
    angle_ki=0.0,
    position_kp=1.11,
    position_kd=2.20,
    integral_limit=5.0,
)


CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE, controller=CONTROLLER)


__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE", "CONTROLLER"]
