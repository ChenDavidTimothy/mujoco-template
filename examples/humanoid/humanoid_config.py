from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    viewer=False,  # True opens the interactive viewer; False runs headless.
    video=False,  # True enables video export using the encoder overrides below.
    logging=True,  # Toggle CSV logging of trajectories; set False to skip log generation entirely.
    simulation_overrides=dict(
        max_steps=6000,  # Positive integer >=1; raise for longer rollouts.
        duration_seconds=6.0,  # None removes the wall-clock limit; otherwise supply any positive float seconds cap.
    ),
    video_overrides=dict(
        path=Path("humanoid_lqr.mp4"),  # Path/str output target (e.g. change extension to .mov or folder prefixes).
        fps=60.0,  # Positive float output frame rate.
        width=1280,  # Positive integer pixel width.
        height=720,  # Positive integer pixel height.
        crf=18,  # Integer 0–51 controlling H.264 quality (lower = better quality).
        preset="medium",  # One of {"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo"}.
        tune=None,  # Optional libx264 tune such as "film","animation","grain","psnr","ssim","fastdecode","zerolatency"; None for default behaviour.
        faststart=True,  # True reorders metadata for streaming; False keeps fastest local export.
        capture_initial_frame=True,  # True saves the pre-sim frame; False starts from the first post-step frame.
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,  # True activates adaptive framing; False leaves the camera static.
            zoom_policy="distance",  # Select "distance", "fov", or "orthographic" to choose zoom mechanism.
            azimuth=120.0,  # Float yaw in degrees (wraps modulo 360).
            elevation=-15.0,  # Float pitch in degrees (-90 straight down, +90 straight up).
            distance=8.0,  # Positive float radius when zoom_policy="distance".
            fovy=None,  # Positive float FOV (degrees) when using "fov"; None defers to defaults.
            ortho_height=None,  # Positive float frame height for orthographic zoom; None uses runtime default.
            lookat=(0.0, 0.0, 1.0),  # Iterable of 3 floats describing the world target point.
            safety_margin=0.2,  # Non-negative float padding multiplier.
            widen_threshold=0.82,  # Float in (0,1) controlling when to zoom out.
            tighten_threshold=0.62,  # Float in (0,1) and < widen_threshold controlling zoom-in triggers.
            smoothing_time_constant=0.35,  # Non-negative float seconds controlling camera responsiveness.
            min_distance=5.0,  # Positive float lower bound for distance zooming.
            max_distance=14.0,  # Positive float >= min_distance bounding farthest distance.
            min_fovy=20.0,  # Positive float lower bound for FOV zooming.
            max_fovy=80.0,  # Positive float >= min_fovy for maximum FOV.
            min_ortho_height=0.5,  # Positive float lower bound for orthographic zooming.
            max_ortho_height=25.0,  # Positive float >= min_ortho_height for maximum orthographic size.
            recenter_axis="x",  # Provide None to disable; otherwise specify axes from {"x","y","z"} (string or iterable).
            recenter_time_constant=1.0,  # Non-negative float seconds for recenter smoothing.
            points_of_interest=(
                "body:torso",
                "body:head",
                "body:pelvis",
                "body:foot_left",
                "body:foot_right",
            ),  # Sequence of MuJoCo tokens ("body:", "site:", "geom:", "bodycom:", "subtreecom:") that drive framing.
        ),
    ),
    viewer_overrides=dict(
        duration_seconds=6.0,  # None leaves viewer running; any positive float auto-closes after that many seconds.
    ),
    logging_overrides=dict(
        path=Path("humanoid_lqr.csv"),  # Path/str output for CSV logs; adjust extension for alternate formats.
        store_rows=True,  # True writes per-step rows; False retains only aggregated statistics.
    ),
)

CONTROLLER = SimpleNamespace(
    keyframe=1,  # Integer index or string name of MuJoCo keyframe used for linearisation.
    height_offset_min_m=-1e-3,  # Float metres lower bound for COM perturbation grid.
    height_offset_max_m=1e-3,  # Float metres upper bound for COM perturbation grid (can exceed min for asymmetric sweeps).
    height_samples=2001,  # Positive integer count of samples across the offset interval.
    linearization_eps=1e-6,  # Positive float perturbation magnitude for numerical Jacobians.
    balance_cost=1000.0,  # Non-negative scalar weighting uprightness error.
    balance_joint_cost=3.0,  # Non-negative scalar weighting balance-critical joint deviations.
    other_joint_cost=0.3,  # Non-negative scalar weighting remaining joints.
    clip_controls=True,  # Boolean: True saturates actuators to limits, False leaves them unconstrained.
    perturbations_enabled=True,  # Boolean toggle for injecting disturbances during training.
    perturb_seed=1,  # Integer RNG seed controlling perturbation sequence reproducibility.
    perturb_duration_s=6.0,  # Positive float seconds covering the perturbation window.
    perturb_ctrl_rate_s=0.8,  # Positive float seconds between perturbation updates.
    perturb_balance_std=0.01,  # Non-negative float standard deviation for balance-directed noise.
    perturb_other_std=0.08,  # Non-negative float standard deviation for other-joint noise.
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "CONTROLLER"]
