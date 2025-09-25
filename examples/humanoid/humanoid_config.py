from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(frozen=True)
class ControllerConfig:
    keyframe: int = 1
    height_offset_min_m: float = -1e-3
    height_offset_max_m: float = 1e-3
    height_samples: int = 2001
    linearization_eps: float = 1e-6
    balance_cost: float = 1000.0
    balance_joint_cost: float = 3.0
    other_joint_cost: float = 0.3
    clip_controls: bool = True


def _default_run_settings() -> mt.PassiveRunSettings:
    return mt.PassiveRunSettings(
        simulation=mt.SimulationSettings(
            max_steps=6000,
            duration_seconds=6.0,
            sample_stride=120,
        ),
        video=mt.VideoSettings(
            enabled=True,
            path=Path("humanoid_lqr.mp4"),
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
        logging=mt.LoggingSettings(
            enabled=False,
            path=Path("humanoid_lqr.csv"),
            store_rows=True,
        ),
    )


@dataclass(frozen=True)
class HumanoidLQRConfig:
    run: mt.PassiveRunSettings = field(default_factory=_default_run_settings)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


CONFIG = HumanoidLQRConfig()


__all__ = [
    "CONFIG",
    "ControllerConfig",
    "HumanoidLQRConfig",
]
