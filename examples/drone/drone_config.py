from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import mujoco_template as mt

from .drone_common import quat_wxyz_from_body_euler


@dataclass(slots=True)
class SimulationOverridesConfig:
    """Simulation loop termination caps."""

    max_steps: int = 4000
    duration_seconds: float | None = 8.0


@dataclass(slots=True)
class VideoOverridesConfig:
    """Parameters controlling encoded video output."""

    path: Path = Path("drone_lqr.mp4")
    fps: float = 60.0
    width: int = 1280
    height: int = 720
    crf: int = 20
    preset: str = "medium"
    tune: str | None = None
    faststart: bool = True
    capture_initial_frame: bool = True
    adaptive_camera: mt.AdaptiveCameraSettings = field(
        default_factory=lambda: mt.AdaptiveCameraSettings(
            enabled=True,
            zoom_policy="distance",
            azimuth=120.0,
            elevation=-15.0,
            distance=4.0,
            fovy=None,
            ortho_height=None,
            lookat=(0.0, 0.0, 1.0),
            safety_margin=0.2,
            widen_threshold=0.82,
            tighten_threshold=0.62,
            smoothing_time_constant=0.05,
            min_distance=4.0,
            max_distance=14.0,
            min_fovy=20.0,
            max_fovy=80.0,
            min_ortho_height=0.5,
            max_ortho_height=25.0,
            recenter_axis="x,y,z",
            recenter_time_constant=0.5,
            points_of_interest=("body:x2",),
        )
    )


@dataclass(slots=True)
class ViewerOverridesConfig:
    """Interactive viewer lifetime configuration."""

    duration_seconds: float | None = 10.0


@dataclass(slots=True)
class LoggingOverridesConfig:
    """CSV logging destinations and retention."""

    path: Path = Path("drone_lqr.csv")
    store_rows: bool = True


@dataclass(slots=True)
class RunSettingsConfig:
    """Top-level runtime switches shown to end users."""

    viewer: bool = False
    video: bool = False
    logging: bool = True
    simulation: SimulationOverridesConfig = field(default_factory=SimulationOverridesConfig)
    video_output: VideoOverridesConfig = field(default_factory=VideoOverridesConfig)
    viewer_settings: ViewerOverridesConfig = field(default_factory=ViewerOverridesConfig)
    logging_settings: LoggingOverridesConfig = field(default_factory=LoggingOverridesConfig)

    def build(self) -> mt.PassiveRunSettings:
        """Materialise a :class:`~mujoco_template.PassiveRunSettings`."""

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
class TrajectoryConfig:
    """Complete specification of the initial and goal state."""

    start_position_m: Tuple[float, float, float] = (0.0, 0.0, 0.7)
    start_orientation_wxyz: Tuple[float, float, float, float] = quat_wxyz_from_body_euler(yaw_deg=0.0)
    start_velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    start_angular_velocity_radps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_position_m: Tuple[float, float, float] = (5.0, -4.0, 2.3)
    goal_orientation_wxyz: Tuple[float, float, float, float] = quat_wxyz_from_body_euler(yaw_deg=180.0)
    goal_velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_angular_velocity_radps: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class ControllerConfig:
    """LQR cost weights and yaw loop shaping."""

    keyframe: str | int = "hover"
    linearization_eps: float = 1e-4
    position_weight: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    orientation_weight: Tuple[float, float, float] = (10.0, 10.0, 20.0)
    velocity_weight: Tuple[float, float, float] = (8.0, 8.0, 6.0)
    angular_velocity_weight: Tuple[float, float, float] = (6.0, 6.0, 10.0)
    control_weight: float = 2.0
    yaw_control_scale: float = 6.0
    yaw_proportional_gain: float = 18.0
    yaw_derivative_gain: float = 4.5
    yaw_integral_gain: float = 4.0
    yaw_integral_limit: float = 6.0
    clip_controls: bool = True
    position_feedback_scale: float | Iterable[float] = 1.0
    orientation_feedback_scale: float | Iterable[float] = 1.0
    velocity_feedback_scale: float | Iterable[float] = 1.0
    angular_velocity_feedback_scale: float | Iterable[float] = 1.0
    goal_position_m: Tuple[float, float, float] = field(default_factory=lambda: TrajectoryConfig().goal_position_m)
    goal_orientation_wxyz: Tuple[float, float, float, float] = field(
        default_factory=lambda: TrajectoryConfig().goal_orientation_wxyz
    )
    goal_velocity_mps: Tuple[float, float, float] = field(default_factory=lambda: TrajectoryConfig().goal_velocity_mps)
    goal_angular_velocity_radps: Tuple[float, float, float] = field(
        default_factory=lambda: TrajectoryConfig().goal_angular_velocity_radps
    )


@dataclass(slots=True)
class ExampleConfig:
    """Aggregated configuration for the drone LQR scenario."""

    run: RunSettingsConfig = field(default_factory=RunSettingsConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


CONFIG = ExampleConfig()

__all__ = [
    "CONFIG",
    "ControllerConfig",
    "ExampleConfig",
    "LoggingOverridesConfig",
    "RunSettingsConfig",
    "SimulationOverridesConfig",
    "TrajectoryConfig",
    "VideoOverridesConfig",
    "ViewerOverridesConfig",
]
