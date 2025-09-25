from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings(
    simulation=mt.SimulationSettings(max_steps=600, duration_seconds=None, sample_stride=120),
    video=mt.VideoSettings(
        enabled=True,
        path=Path("pendulum_passive.mp4"),
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
            azimuth=100.0,
            elevation=-18.0,
            distance=3.0,
            lookat=(0.0, 0.0, -0.3),
            min_distance=2.0,
            max_distance=5.0,
            safety_margin=0.1,
            widen_threshold=0.85,
            tighten_threshold=0.65,
            smoothing_time_constant=0.22,
            recenter_axis="x",
            recenter_time_constant=1.1,
            points_of_interest=("body:arm", "site:tip"),
        ),
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("pendulum_passive.csv"), store_rows=True),
)

INITIAL_STATE = SimpleNamespace(angle_deg=90.0, velocity_deg=0.0)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE)

__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE"]
