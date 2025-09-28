from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    viewer=False,  # True opens the interactive viewer; False stays headless.
    video=False,  # True exports video using the following encoder configuration.
    logging=True,  # Toggle CSV logging of state/control; False disables logging entirely.
    simulation_overrides=dict(
        max_steps=600,  # Positive integer >=1; extend for longer simulations before force-stop.
        duration_seconds=None,  # None removes the wall-clock cap; otherwise provide any positive float seconds limit.
    ),
    video_overrides=dict(
        path=Path("pendulum_passive.mp4"),  # Path/str for the exported movie (change directories or extensions freely).
        fps=60.0,  # Positive float encoded frame rate.
        width=1280,  # Positive integer pixel width.
        height=720,  # Positive integer pixel height.
        crf=18,  # Integer 0–51 controlling H.264 quality (lower numbers increase fidelity).
        preset="medium",  # Select from {"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo"} to trade speed for compression.
        tune=None,  # Optional tune flag such as "film","animation","grain","psnr","ssim","fastdecode","zerolatency"; None for defaults.
        faststart=True,  # True optimises MP4 for streaming; False leaves atoms in original order.
        capture_initial_frame=True,  # True records the t=0 frame; False begins after the first simulation step.
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,  # True engages adaptive framing; False keeps the camera static.
            zoom_policy="distance",  # Choose "distance", "fov", or "orthographic" for zoom behaviour.
            azimuth=100.0,  # Float yaw angle in degrees.
            elevation=-18.0,  # Float pitch angle in degrees.
            distance=3.0,  # Positive float radius when zooming by distance.
            fovy=None,  # Positive float FOV (degrees) when zoom_policy="fov"; None lets runtime choose.
            ortho_height=None,  # Positive float orthographic window height; None keeps defaults.
            lookat=(0.0, 0.0, -0.3),  # Iterable of 3 floats specifying the camera's target point.
            safety_margin=0.1,  # Non-negative float padding multiplier.
            widen_threshold=0.85,  # Float in (0,1) determining zoom-out trigger.
            tighten_threshold=0.65,  # Float in (0,1) less than widen_threshold triggering zoom-in.
            smoothing_time_constant=0.22,  # Non-negative float seconds smoothing the camera response.
            min_distance=2.0,  # Positive float minimum distance when zooming by distance.
            max_distance=5.0,  # Positive float >= min_distance bounding maximum distance.
            min_fovy=20.0,  # Positive float minimum FOV for FOV zooming.
            max_fovy=80.0,  # Positive float >= min_fovy maximum FOV.
            min_ortho_height=0.5,  # Positive float minimum orthographic extent.
            max_ortho_height=25.0,  # Positive float >= min_ortho_height maximum orthographic extent.
            recenter_axis="x",  # None disables recentering; otherwise choose axes from {"x","y","z"} via string or iterable.
            recenter_time_constant=1.1,  # Non-negative float seconds for recenter smoothing.
            points_of_interest=("body:arm", "site:tip"),  # Sequence of tokens ("body:","site:","geom:","bodycom:","subtreecom:") to track.
        ),
    ),
    viewer_overrides=dict(
        duration_seconds=5.0,  # None keeps the viewer open indefinitely; positive float closes after that time.
    ),
    logging_overrides=dict(
        path=Path("pendulum_passive.csv"),  # Path/str destination for CSV logs; change extension for other formats.
        store_rows=True,  # True writes every stored row; False keeps only aggregated metrics.
    ),
)

INITIAL_STATE = SimpleNamespace(
    angle_deg=90.0,  # Any float degrees displacement from upright (positive rotates counter-clockwise).
    velocity_deg=0.0,  # Float deg/s initial angular velocity.
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, initial_state=INITIAL_STATE)

__all__ = ["CONFIG", "RUN_SETTINGS", "INITIAL_STATE"]
