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
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("pendulum_passive.csv"), store_rows=True),
)

INITIAL_STATE = SimpleNamespace(angle_deg=90.0, velocity_deg=0.0)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE)

__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE"]
