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
    path: Path = Path("cartpole.mp4")
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
    path: Path = Path("cartpole.csv")


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


@dataclass(frozen=True)
class SimulationConfig:
    headless_max_steps: int = 2000
    headless_duration_seconds: float | None = None
    sample_stride: int = 50


@dataclass(frozen=True)
class CartPoleConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


CONFIG = CartPoleConfig()


__all__ = [
    "CONFIG",
    "CartPoleConfig",
    "ControllerConfig",
    "InitialStateConfig",
    "LoggingConfig",
    "SimulationConfig",
    "VideoConfig",
    "ViewerConfig",
]
