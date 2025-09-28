from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    viewer=False,  # True opens the interactive viewer; False executes headless.
    video=False,  # True enables encoded video export with the below parameters.
    logging=True,  # Toggle CSV trajectory logging; False skips log generation.
    simulation_overrides=dict(
        max_steps=400,  # Positive integer >=1 controlling the hard stop on physics steps.
        duration_seconds=None,  # None for unlimited wall-clock time; any positive float caps runtime in seconds.
    ),
    video_overrides=dict(
        path=Path("pendulum.mp4"),  # Path/str for the exported clip (use different extensions or directories as desired).
        fps=60.0,  # Positive float encoded frame rate.
        width=1280,  # Positive integer pixel width.
        height=720,  # Positive integer pixel height.
        crf=18,  # Integer 0–51 quality knob (lower = better quality).
        preset="medium",  # Choose from {"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo"} to trade speed vs. compression.
        tune=None,  # Optional libx264 tune: "film","animation","grain","psnr","ssim","fastdecode","zerolatency"; None uses defaults.
        faststart=True,  # True optimises MP4 for streaming; False avoids the extra mux step.
        capture_initial_frame=True,  # True writes the t=0 frame; False starts after the first physics step.
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,  # True enables adaptive framing; False leaves the camera fixed.
            zoom_policy="distance",  # Select "distance", "fov", or "orthographic" to set zoom mode.
            azimuth=90.0,  # Float yaw angle in degrees.
            elevation=-20.0,  # Float pitch angle in degrees.
            distance=2.5,  # Positive float radius when zooming by distance.
            fovy=None,  # Positive float FOV (degrees) when using "fov" policy; None defers to defaults.
            ortho_height=None,  # Positive float orthographic window height; None keeps runtime default.
            lookat=(0.0, 0.0, -0.25),  # Iterable of 3 floats specifying the world-space target.
            safety_margin=0.08,  # Non-negative float padding multiplier around tracked points.
            widen_threshold=0.8,  # Float in (0,1) that triggers zoom-out.
            tighten_threshold=0.6,  # Float in (0,1) < widen_threshold triggering zoom-in.
            smoothing_time_constant=0.18,  # Non-negative float seconds controlling responsiveness.
            min_distance=1.5,  # Positive float lower bound for distance zoom.
            max_distance=4.5,  # Positive float >= min_distance for maximum pull-back.
            min_fovy=20.0,  # Positive float lower FOV bound when zooming by FOV.
            max_fovy=80.0,  # Positive float >= min_fovy for maximum FOV.
            min_ortho_height=0.5,  # Positive float minimum orthographic extent.
            max_ortho_height=25.0,  # Positive float >= min_ortho_height for maximum orthographic extent.
            recenter_axis="x",  # None disables recentering; otherwise choose axes from {"x","y","z"} via string or iterable.
            recenter_time_constant=0.9,  # Non-negative float seconds smoothing recenter motions.
            points_of_interest=("body:arm", "site:tip"),  # Sequence of tokens ("body:","site:","geom:","bodycom:","subtreecom:") the camera follows.
        ),
    ),
    viewer_overrides=dict(
        duration_seconds=5.0,  # None keeps viewer open; positive float auto-exits after that many seconds.
    ),
    logging_overrides=dict(
        path=Path("pendulum.csv"),  # Path/str destination for logs; adjust extension for alternative formats.
        store_rows=True,  # True records every stored step row; False retains only aggregates.
    ),
)

INITIAL_STATE = SimpleNamespace(
    angle_deg=60.0,  # Any float degrees offset from upright (positive counter-clockwise).
    velocity_deg=0.0,  # Float deg/s initial angular velocity.
)

CONTROLLER = SimpleNamespace(
    kp=20.0,  # Non-negative proportional gain on angle error.
    kd=4.0,  # Non-negative derivative gain on angular velocity.
    target_angle_deg=0.0,  # Float degrees specifying the desired equilibrium (e.g. 180 for inverted).
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE", "CONTROLLER"]
