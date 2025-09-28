from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco_template as mt


@dataclass(slots=True)
class SimulationOverridesConfig:
    """Simulation limits for the humanoid balance task."""

    max_steps: int = 6000
    duration_seconds: float | None = 6.0


@dataclass(slots=True)
class VideoOverridesConfig:
    """Video encoder settings for the humanoid rollout."""

    path: Path = Path("humanoid_lqr.mp4")
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
            azimuth=120.0,
            elevation=-15.0,
            distance=8.0,
            fovy=None,
            ortho_height=None,
            lookat=(0.0, 0.0, 1.0),
            safety_margin=0.2,
            widen_threshold=0.82,
            tighten_threshold=0.62,
            smoothing_time_constant=0.35,
            min_distance=5.0,
            max_distance=14.0,
            min_fovy=20.0,
            max_fovy=80.0,
            min_ortho_height=0.5,
            max_ortho_height=25.0,
            recenter_axis="x",
            recenter_time_constant=1.0,
            points_of_interest=(
                "body:torso",
                "body:head",
                "body:pelvis",
                "body:foot_left",
                "body:foot_right",
            ),
        )
    )


@dataclass(slots=True)
class ViewerOverridesConfig:
    """Interactive viewer options."""

    duration_seconds: float | None = 6.0


@dataclass(slots=True)
class LoggingOverridesConfig:
    """Logging targets for the humanoid example."""

    path: Path = Path("humanoid_lqr.csv")
    store_rows: bool = True


@dataclass(slots=True)
class RunSettingsConfig:
    """High-level execution controls for the humanoid harness."""

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
class ControllerConfig:
    """Parameters for humanoid LQR balance."""

    keyframe: int = 1
    height_offset_min_m: float = -1e-3
    height_offset_max_m: float = 1e-3
    height_samples: int = 2001
    linearization_eps: float = 1e-6
    balance_cost: float = 1000.0
    balance_joint_cost: float = 3.0
    other_joint_cost: float = 0.3
    clip_controls: bool = True
    perturbations_enabled: bool = True
    perturb_seed: int = 1
    perturb_duration_s: float = 6.0
    perturb_ctrl_rate_s: float = 0.8
    perturb_balance_std: float = 0.01
    perturb_other_std: float = 0.08


@dataclass(slots=True)
class ExampleConfig:
    """Aggregated configuration for the humanoid LQR scenario."""

    run: RunSettingsConfig = field(default_factory=RunSettingsConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


CONFIG = ExampleConfig()

__all__ = [
    "CONFIG",
    "ControllerConfig",
    "ExampleConfig",
    "LoggingOverridesConfig",
    "RunSettingsConfig",
    "SimulationOverridesConfig",
    "VideoOverridesConfig",
    "ViewerOverridesConfig",
]
