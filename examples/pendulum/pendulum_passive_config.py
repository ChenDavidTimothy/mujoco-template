from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco_template as mt


def _default_run_settings() -> mt.PassiveRunSettings:
    return mt.PassiveRunSettings(
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


@dataclass(frozen=True)
class InitialStateConfig:
    angle_deg: float = 90.0
    velocity_deg: float = 0.0


@dataclass(frozen=True)
class PendulumPassiveConfig:
    run: mt.PassiveRunSettings = field(default_factory=_default_run_settings)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)


CONFIG = PendulumPassiveConfig()


__all__ = [
    "CONFIG",
    "InitialStateConfig",
    "PendulumPassiveConfig",
]
