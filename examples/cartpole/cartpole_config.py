from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt


RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    viewer=False,  # Set True to launch the interactive viewer; keep False for headless runs.
    video=False,  # Set True to enable MP4 export with the below encoder settings.
    logging=True,  # Toggle CSV logging of state/action histories (False disables logging entirely).
    simulation_overrides=dict(
        max_steps=2000,  # Positive integer >=1; increase for longer episodes before forced termination.
        duration_seconds=None,  # Either None for unlimited wall-clock time or any positive float cap (seconds).
        sample_stride=50,  # Positive integer >=1 controlling log decimation; 1 records every step, higher skips samples.
    ),
    video_overrides=dict(
        path=Path("cartpole.mp4"),  # Any pathlib.Path or string target (e.g. ".mov", nested directories, network mounts).
        fps=60.0,  # Positive float playback rate; match sim for real-time or lower/raise for slow/fast motion.
        width=1280,  # Positive integer pixel width; pick 1920 for 1080p, 3840 for 4K, etc.
        height=720,  # Positive integer pixel height; pair with width to control aspect/resolution.
        crf=18,  # Integer 0–51 for libx264 quality (0=lossless, 18≈visually lossless, 51=worst compression).
        preset="medium",  # One of FFmpeg's speed presets: "ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo".
        tune=None,  # Optional libx264 tune such as "film","animation","grain","psnr","ssim","fastdecode","zerolatency"; None disables tuning.
        faststart=True,  # True moves metadata for web streaming; set False to skip the extra muxing step.
        capture_initial_frame=True,  # True writes the t=0 frame before stepping; False omits it for strictly dynamic footage.
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,  # True activates automatic framing; False keeps the default static camera.
            zoom_policy="distance",  # Choose from {"distance","fov","orthographic"} to control zoom mode.
            azimuth=90.0,  # Any float degrees yaw around the scene (wraps modulo 360).
            elevation=-20.0,  # Float degrees pitch; -90 looks straight down, +90 straight up.
            distance=1.0,  # Positive float meters when using distance zooming; shrink for closer shots.
            fovy=None,  # Positive float field of view in degrees when zoom_policy="fov"; leave None to use defaults.
            ortho_height=None,  # Positive float extent when zoom_policy="orthographic"; None keeps last value.
            lookat=(0.0, 0.0, 0.0),  # Iterable of 3 floats world target; supply another body/site COM to follow new focus.
            safety_margin=0.1,  # Non-negative float padding scale; raise to leave extra headroom around subjects.
            widen_threshold=0.75,  # Float in (0,1); higher delays zoom-out triggers.
            tighten_threshold=0.55,  # Float in (0,1) and < widen_threshold; lower tightens sooner.
            smoothing_time_constant=0.2,  # Non-negative float seconds; raise for smoother camera response.
            min_distance=1.0,  # Positive float lower bound for distance zooming.
            max_distance=12.0,  # Positive float >= min_distance; expand for wider travel range.
            min_fovy=20.0,  # Positive float lower bound for FOV zooming.
            max_fovy=80.0,  # Positive float >= min_fovy; increase to allow wider angles.
            min_ortho_height=0.5,  # Positive float lower bound for orthographic zooming.
            max_ortho_height=25.0,  # Positive float >= min_ortho_height; expand for taller framing.
            recenter_axis="x",  # None disables recentering; otherwise choose "x","y","z" or any iterable/comma string of axes.
            recenter_time_constant=1.5,  # Non-negative float seconds; larger values recenter more slowly.
            points_of_interest=(
                "body:cart",
                "site:tip",
            ),  # Sequence of MuJoCo tokens: "body:<name>", "site:<name>", "geom:<name>", "bodycom:<name>", or "subtreecom:<name>".
        ),
    ),
    viewer_overrides=dict(
        duration_seconds=6.0,  # None keeps the viewer open indefinitely; any positive float auto-closes after that many seconds.
    ),
    logging_overrides=dict(
        path=Path("cartpole.csv"),  # Path or string output (e.g. JSON, Parquet) for logs; directories created automatically.
        store_rows=True,  # True writes per-step rows; set False to retain only summary statistics in memory.
    ),
)


INITIAL_STATE = SimpleNamespace(
    cart_position=0.0,  # Any float meters left/right displacement for initial cart placement.
    cart_velocity=0.0,  # Float m/s initial cart velocity; use non-zero to study transient corrections.
    pole_angle_deg=30.0,  # Float degrees from vertical (+ tilts forward, - backward); ±180 allowed.
    pole_velocity_deg=0.0,  # Float deg/s initial angular velocity of the pole.
)


CONTROLLER = SimpleNamespace(
    angle_kp=16.66,  # Non-negative proportional gain on pole angle error; larger stiffens correction.
    angle_kd=4.45,  # Non-negative derivative gain on pole angular velocity for damping.
    angle_ki=0.0,  # Integral gain on angle; >0 combats bias but risks windup.
    position_kp=1.11,  # Non-negative proportional gain on cart position error.
    position_kd=2.20,  # Non-negative derivative gain on cart velocity.
    integral_limit=5.0,  # Positive clamp magnitude on integral accumulator to prevent windup.
)


CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE, controller=CONTROLLER)


__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE", "CONTROLLER"]
