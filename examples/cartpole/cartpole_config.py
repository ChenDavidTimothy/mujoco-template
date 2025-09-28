from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(slots=True)
class SimulationOverridesConfig:
    """Termination criteria for the cartpole rollout."""

    max_steps: int = 2000
    duration_seconds: float | None = None


@dataclass(slots=True)
class VideoOverridesConfig:
    """Encoded video export settings exposed to researchers."""

    path: Path = Path("cartpole.mp4")
    fps: float = 60.0
    width: int = 1280
    height: int = 720
    crf: int = 18
    preset: str = "medium"
    tune: str | None = None
    faststart: bool = True
    capture_initial_frame: bool = True
    adaptive_camera: mt.AdaptiveCameraSettings = field(
        default_factory=lambda: mt.AdaptiveCameraSettings(
            enabled=True,
            zoom_policy="distance",
            azimuth=90.0,
            elevation=-20.0,
            distance=1.0,
            fovy=None,
            ortho_height=None,
            lookat=(0.0, 0.0, 0.0),
            safety_margin=0.1,
            widen_threshold=0.75,
            tighten_threshold=0.55,
            smoothing_time_constant=0.2,
            min_distance=1.0,
            max_distance=12.0,
            min_fovy=20.0,
            max_fovy=80.0,
            min_ortho_height=0.5,
            max_ortho_height=25.0,
            recenter_axis="x",
            recenter_time_constant=1.5,
            points_of_interest=("body:cart", "site:tip"),
        )
    )


@dataclass(slots=True)
class ViewerOverridesConfig:
    """Interactive viewer configuration."""

    duration_seconds: float | None = 6.0


@dataclass(slots=True)
class LoggingOverridesConfig:
    """Logging destination and sampling behaviour."""

    path: Path = Path("cartpole.csv")
    store_rows: bool = True


@dataclass(slots=True)
class RunSettingsConfig:
    """High-level execution toggles for the cartpole experiment."""

    viewer: bool = False
    video: bool = False
    logging: bool = True
    simulation: SimulationOverridesConfig = field(default_factory=SimulationOverridesConfig)
    video_output: VideoOverridesConfig = field(default_factory=VideoOverridesConfig)
    viewer_settings: ViewerOverridesConfig = field(default_factory=ViewerOverridesConfig)
    logging_settings: LoggingOverridesConfig = field(default_factory=LoggingOverridesConfig)

    def build(self) -> mt.PassiveRunSettings:
        """Produce a concrete :class:`~mujoco_template.PassiveRunSettings`."""

        return mt.PassiveRunSettings.from_flags(
            viewer=self.viewer,
            video=self.video,
            logging=self.logging,
            simulation_overrides=asdict(self.simulation),
            video_overrides={**asdict(self.video_output), "adaptive_camera": self.video_output.adaptive_camera},
            viewer_overrides=asdict(self.viewer_settings),
            logging_overrides=asdict(self.logging_settings),
        )


@dataclass(slots=True)
class InitialStateConfig:
    """Initial cart and pole state specification."""

    cart_position: float = 0.0
    cart_velocity: float = 0.0
    pole_angle_deg: float = 30.0
    pole_velocity_deg: float = 0.0


@dataclass(slots=True)
class ControllerConfig:
    """PID balance gains for the cartpole."""

    angle_kp: float = 16.66
    angle_kd: float = 4.45
    angle_ki: float = 0.0
    position_kp: float = 1.11
    position_kd: float = 2.20
    integral_limit: float = 5.0


@dataclass(slots=True)
class ExampleConfig:
    """Aggregated configuration for the cartpole PID example."""

    run: RunSettingsConfig = field(default_factory=RunSettingsConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


CONFIG = ExampleConfig()

__all__ = [
    "CONFIG",
    "ControllerConfig",
    "ExampleConfig",
    "InitialStateConfig",
    "LoggingOverridesConfig",
    "RunSettingsConfig",
    "SimulationOverridesConfig",
    "VideoOverridesConfig",
    "ViewerOverridesConfig",
]
