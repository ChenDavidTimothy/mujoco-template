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
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("pendulum.csv"), store_rows=True),
)

INITIAL_STATE = SimpleNamespace(angle_deg=60.0, velocity_deg=0.0)

CONTROLLER = SimpleNamespace(kp=20.0, kd=4.0, target_angle_deg=0.0)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE", "CONTROLLER"]
