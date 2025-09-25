from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings(
    simulation=mt.SimulationSettings(max_steps=400, duration_seconds=None, sample_stride=80),
    video=mt.VideoSettings(
        enabled=True,
        path=Path("pendulum.mp4"),
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
            distance=2.5,
            lookat=(0.0, 0.0, -0.25),
            min_distance=1.5,
            max_distance=4.5,
            safety_margin=0.08,
            widen_threshold=0.8,
            tighten_threshold=0.6,
            smoothing_time_constant=0.18,
            recenter_axis="x",
            recenter_time_constant=0.9,
            points_of_interest=("body:arm", "site:tip"),
        ),
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("pendulum.csv"), store_rows=True),
)

INITIAL_STATE = SimpleNamespace(angle_deg=60.0, velocity_deg=0.0)

CONTROLLER = SimpleNamespace(kp=20.0, kd=4.0, target_angle_deg=0.0)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE", "CONTROLLER"]
