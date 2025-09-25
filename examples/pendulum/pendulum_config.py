from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ViewerConfig:
    enabled: bool = False
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VideoConfig:
    enabled: bool = True
    path: Path = Path("pendulum.mp4")
    fps: float = 60.0
    width: int = 1280
    height: int = 720
    crf: int = 18
    preset: str = "medium"
    tune: str | None = None
    faststart: bool = True
    capture_initial_frame: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = False
    path: Path = Path("pendulum.csv")


@dataclass(frozen=True)
class ControllerConfig:
    kp: float = 20.0
    kd: float = 4.0
    target_angle_deg: float = 0.0


@dataclass(frozen=True)
class InitialStateConfig:
    angle_deg: float = 60.0
    velocity_deg: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    headless_max_steps: int = 400
    headless_duration_seconds: float | None = None
    sample_stride: int = 80


@dataclass(frozen=True)
class PendulumConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


CONFIG = PendulumConfig()


__all__ = [
    "CONFIG",
    "ControllerConfig",
    "InitialStateConfig",
    "LoggingConfig",
    "PendulumConfig",
    "SimulationConfig",
    "VideoConfig",
    "ViewerConfig",
]
