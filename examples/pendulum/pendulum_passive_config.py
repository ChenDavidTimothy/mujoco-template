from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(slots=True)
class SimulationOverridesConfig:
    """Simulation caps for the passive pendulum swing."""

    max_steps: int = 600
    duration_seconds: float | None = None


@dataclass(slots=True)
class VideoOverridesConfig:
    """Video encoder settings for the passive rollout."""

    path: Path = Path("pendulum_passive.mp4")
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
            azimuth=100.0,
            elevation=-18.0,
            distance=3.0,
            fovy=None,
            ortho_height=None,
            lookat=(0.0, 0.0, -0.3),
            safety_margin=0.1,
            widen_threshold=0.85,
            tighten_threshold=0.65,
            smoothing_time_constant=0.22,
            min_distance=2.0,
            max_distance=5.0,
            min_fovy=20.0,
            max_fovy=80.0,
            min_ortho_height=0.5,
            max_ortho_height=25.0,
            recenter_axis="x",
            recenter_time_constant=1.1,
            points_of_interest=("body:arm", "site:tip"),
        )
    )


@dataclass(slots=True)
class ViewerOverridesConfig:
    """Interactive viewer runtime."""

    duration_seconds: float | None = 5.0


@dataclass(slots=True)
class LoggingOverridesConfig:
    """Logging configuration for passive swings."""

    path: Path = Path("pendulum_passive.csv")
    store_rows: bool = True


@dataclass(slots=True)
class RunSettingsConfig:
    """Top-level toggles for the passive pendulum harness."""

    viewer: bool = False
    video: bool = False
    logging: bool = True
    simulation: SimulationOverridesConfig = field(default_factory=SimulationOverridesConfig)
    video_output: VideoOverridesConfig = field(default_factory=VideoOverridesConfig)
    viewer_settings: ViewerOverridesConfig = field(default_factory=ViewerOverridesConfig)
    logging_settings: LoggingOverridesConfig = field(default_factory=LoggingOverridesConfig)

    def build(self) -> mt.PassiveRunSettings:
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
    """Initial angle and velocity for passive dynamics."""

    angle_deg: float = 90.0
    velocity_deg: float = 0.0


@dataclass(slots=True)
class ExampleConfig:
    """Aggregated configuration for the passive pendulum example."""

    run: RunSettingsConfig = field(default_factory=RunSettingsConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)


CONFIG = ExampleConfig()

__all__ = [
    "CONFIG",
    "ExampleConfig",
    "InitialStateConfig",
    "LoggingOverridesConfig",
    "RunSettingsConfig",
    "SimulationOverridesConfig",
    "VideoOverridesConfig",
    "ViewerOverridesConfig",
]
