from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(frozen=True)
class ControllerConfig:
    kp: float = 20.0
    kd: float = 4.0
    target_angle_deg: float = 0.0


@dataclass(frozen=True)
class InitialStateConfig:
    angle_deg: float = 60.0
    velocity_deg: float = 0.0


def _default_run_settings() -> mt.PassiveRunSettings:
    return mt.PassiveRunSettings(
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


@dataclass(frozen=True)
class PendulumConfig:
    run: mt.PassiveRunSettings = field(default_factory=_default_run_settings)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


CONFIG = PendulumConfig()


__all__ = [
    "CONFIG",
    "ControllerConfig",
    "InitialStateConfig",
    "PendulumConfig",
]
