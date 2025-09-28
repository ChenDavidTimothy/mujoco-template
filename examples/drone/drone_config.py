from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

from drone_common import quat_wxyz_from_body_euler

RUN_SETTINGS = mt.PassiveRunSettings.from_flags(
    viewer=False,  # True launches the interactive viewer; False runs headless.
    video=False,  # True enables encoded video export with the below parameters.
    logging=True,  # Toggle per-step CSV logging; set False when logs are unnecessary.
    simulation_overrides=dict(
        max_steps=4000,  # Positive integer >=1; bump upward for longer flights before forced stop.
        duration_seconds=8.0,  # Either None for unlimited wall-clock time or any positive float seconds cap.
        sample_stride=80,  # Positive integer >=1; lower numbers store denser telemetry, higher sparsify logs.
    ),
    video_overrides=dict(
        path=Path("drone_lqr.mp4"),  # Any Path/str target (e.g. "outputs/run1.mp4" or alternate container extensions).
        fps=60.0,  # Positive float output frame rate; raise for slow motion capture, lower to shrink files.
        width=1280,  # Positive integer pixel width; 1920 or 3840 for HD/UHD, etc.
        height=720,  # Positive integer pixel height controlling vertical resolution.
        crf=20,  # Integer 0–51 (libx264 quality); lower improves fidelity, higher boosts compression.
        preset="medium",  # One of {"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo"}.
        tune=None,  # Optional libx264 tune such as "film","animation","grain","psnr","ssim","fastdecode","zerolatency"; None for default heuristics.
        faststart=True,  # True relocates metadata for streaming; False leaves the MP4 optimized for local playback only.
        capture_initial_frame=True,  # True saves the pre-step frame; False starts after the first step is applied.
        adaptive_camera=mt.AdaptiveCameraSettings(
            enabled=True,  # True activates adaptive framing; False keeps a fixed viewpoint.
            zoom_policy="distance",  # Select from {"distance","fov","orthographic"} depending on zoom semantics.
            azimuth=120.0,  # Float degrees yaw orientation (wraps modulo 360).
            elevation=-15.0,  # Float degrees pitch; [-90, 90] covers nadir-to-zenith.
            distance=4.0,  # Positive float radius when zooming by distance.
            fovy=None,  # Positive float FOV (degrees) when using "fov" policy; None keeps default.
            ortho_height=None,  # Positive float extent for orthographic policy; None defers to runtime default.
            lookat=(0.0, 0.0, 1.0),  # 3-float iterable world target; retarget to another body/site token.
            safety_margin=0.2,  # Non-negative float padding multiplier to keep extra space around subjects.
            widen_threshold=0.82,  # Float in (0,1) controlling zoom-out hysteresis.
            tighten_threshold=0.62,  # Float in (0,1) strictly less than widen_threshold to trigger zoom-in.
            smoothing_time_constant=0.05,  # Non-negative float seconds; bigger -> smoother, smaller -> more responsive.
            min_distance=4.0,  # Positive float lower distance bound.
            max_distance=14.0,  # Positive float >= min_distance controlling farthest pull-back.
            min_fovy=20.0,  # Positive float lower FOV bound.
            max_fovy=80.0,  # Positive float >= min_fovy for maximum wide angle.
            min_ortho_height=0.5,  # Positive float lower orthographic bound.
            max_ortho_height=25.0,  # Positive float >= min_ortho_height for widest ortho window.
            recenter_axis="x,y,z",  # Provide None to disable; otherwise string/iterable combination of axes from {"x","y","z"}.
            recenter_time_constant=0.5,  # Non-negative float seconds for recenter smoothing.
            points_of_interest=("body:x2",),  # Sequence of tokens ("body:","site:","geom:","bodycom:","subtreecom:") naming items to track.
        ),
    ),
    viewer_overrides=dict(
        duration_seconds=10.0,  # None leaves the viewer open; any positive float auto-closes after that wall-clock time.
    ),
    logging_overrides=dict(
        path=Path("drone_lqr.csv"),  # Path/str destination for CSV logs; change extension for different formats.
        store_rows=True,  # True writes per-step rows; False keeps only aggregates to shrink disk usage.
    ),
)

TRAJECTORY = SimpleNamespace(
    start_position_m=(0.0, 0.0, 0.7),  # Tuple of floats metres; change to reposition the drone at t=0.
    start_orientation_wxyz=quat_wxyz_from_body_euler(yaw_deg=0.0),  # Unit quaternion (w,x,y,z) start attitude; any valid orientation.
    start_velocity_mps=(0.0, 0.0, 0.0),  # Tuple floats m/s initial linear velocity.
    start_angular_velocity_radps=(0.0, 0.0, 0.0),  # Tuple floats rad/s initial angular rates.
    goal_position_m=(5.0, -4.0, 2.3),  # Tuple floats metres specifying terminal position target.
    goal_orientation_wxyz=quat_wxyz_from_body_euler(yaw_deg=180.0),  # Unit quaternion goal attitude.
    goal_velocity_mps=(0.0, 0.0, 0.0),  # Desired final linear velocity vector in m/s.
    goal_angular_velocity_radps=(0.0, 0.0, 0.0),  # Desired final angular velocity in rad/s.
)

CONTROLLER = SimpleNamespace(
    keyframe="hover",  # Name of a MuJoCo keyframe to linearise about; choose any defined keyframe label.
    linearization_eps=1e-4,  # Positive float perturbation size for finite differences; shrink for higher fidelity.
    position_weight=(10.0, 10.0, 10.0),  # Tuple of non-negative weights on xyz position error in the LQR cost.
    orientation_weight=(10.0, 10.0, 20.0),  # Tuple of non-negative weights for orientation error components.
    velocity_weight=(8.0, 8.0, 6.0),  # Tuple of non-negative weights on linear velocity error.
    angular_velocity_weight=(6.0, 6.0, 10.0),  # Tuple of non-negative weights on angular velocity error.
    control_weight=2.0,  # Non-negative scalar penalising control effort.
    yaw_control_scale=6.0,  # Positive scalar multiplier for yaw commands.
    yaw_proportional_gain=18.0,  # Non-negative P gain for the yaw outer loop.
    yaw_derivative_gain=4.5,  # Non-negative D gain for yaw damping.
    yaw_integral_gain=4.0,  # Non-negative I gain; zero disables integral action.
    yaw_integral_limit=6.0,  # Positive clamp magnitude to prevent yaw integral windup.
    clip_controls=True,  # Boolean: True enforces actuator saturation, False allows unconstrained commands.
    goal_position_m=TRAJECTORY.goal_position_m,  # Reference position tuple; override to chase a new target.
    goal_orientation_wxyz=TRAJECTORY.goal_orientation_wxyz,  # Reference orientation quaternion for the controller.
    goal_velocity_mps=TRAJECTORY.goal_velocity_mps,  # Reference terminal linear velocity vector.
    goal_angular_velocity_radps=TRAJECTORY.goal_angular_velocity_radps,  # Reference terminal angular velocity vector.
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, trajectory=TRAJECTORY, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "TRAJECTORY", "CONTROLLER"]
