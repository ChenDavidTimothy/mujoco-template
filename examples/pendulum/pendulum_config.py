from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(slots=True)
class SimulationOverridesConfig:
    """Physics loop termination settings for the controlled pendulum."""

    max_steps: int = 400
    duration_seconds: float | None = None


@dataclass(slots=True)
class VideoOverridesConfig:
    """Encoded video parameters for the PD-controlled pendulum."""

    path: Path = Path("pendulum.mp4")
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
            distance=2.5,
            fovy=None,
            ortho_height=None,
            lookat=(0.0, 0.0, -0.25),
            safety_margin=0.08,
            widen_threshold=0.8,
            tighten_threshold=0.6,
            smoothing_time_constant=0.18,
            min_distance=1.5,
            max_distance=4.5,
            min_fovy=20.0,
            max_fovy=80.0,
            min_ortho_height=0.5,
            max_ortho_height=25.0,
            recenter_axis="x",
            recenter_time_constant=0.9,
            points_of_interest=("body:arm", "site:tip"),
        )
    )


@dataclass(slots=True)
class ViewerOverridesConfig:
    """Interactive viewer behaviour."""

    duration_seconds: float | None = 5.0


@dataclass(slots=True)
class LoggingOverridesConfig:
    """Per-run logging configuration."""

    path: Path = Path("pendulum.csv")
    store_rows: bool = True


@dataclass(slots=True)
class RunSettingsConfig:
    """Switches and overrides for the PD pendulum harness."""

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
    """Initial pendulum pose in degrees."""

    angle_deg: float = 60.0
    velocity_deg: float = 0.0


@dataclass(slots=True)
class ControllerConfig:
    """PD gains and target for the upright pendulum."""

    kp: float = 20.0
    kd: float = 4.0
    target_angle_deg: float = 0.0


@dataclass(slots=True)
class ExampleConfig:
    """Aggregated configuration for the controlled pendulum example."""

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
