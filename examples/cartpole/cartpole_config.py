from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(frozen=True)
class ControllerConfig:
    angle_kp: float = 16.66
    angle_kd: float = 4.45
    angle_ki: float = 0.0
    position_kp: float = 1.11
    position_kd: float = 2.20
    integral_limit: float = 5.0


@dataclass(frozen=True)
class InitialStateConfig:
    cart_position: float = 0.0
    cart_velocity: float = 0.0
    pole_angle_deg: float = 30.0
    pole_velocity_deg: float = 0.0


def _default_run_settings() -> mt.PassiveRunSettings:
    return mt.PassiveRunSettings(
        simulation=mt.SimulationSettings(max_steps=2000, duration_seconds=None, sample_stride=50),
        video=mt.VideoSettings(
            enabled=True,
            path=Path("cartpole.mp4"),
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
        logging=mt.LoggingSettings(enabled=False, path=Path("cartpole.csv"), store_rows=True),
    )


@dataclass(frozen=True)
class CartPoleConfig:
    run: mt.PassiveRunSettings = field(default_factory=_default_run_settings)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


CONFIG = CartPoleConfig()


__all__ = [
    "CONFIG",
    "CartPoleConfig",
    "ControllerConfig",
    "InitialStateConfig",
]
