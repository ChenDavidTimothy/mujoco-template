from pathlib import Path
from types import SimpleNamespace

import mujoco_template as mt

RUN_SETTINGS = mt.PassiveRunSettings(
    simulation=mt.SimulationSettings(max_steps=6000, duration_seconds=6.0, sample_stride=120),
    video=mt.VideoSettings(
        enabled=True,
        path=Path("humanoid_lqr.mp4"),
        fps=60.0,
        width=1280,
        height=720,
        crf=18,
        preset="medium",
        tune=None,
        faststart=True,
        capture_initial_frame=True,
    ),
    viewer=mt.ViewerSettings(enabled=False, duration_seconds=None),
    logging=mt.LoggingSettings(enabled=False, path=Path("humanoid_lqr.csv"), store_rows=True),
)

CONTROLLER = SimpleNamespace(
    keyframe=1,
    height_offset_min_m=-1e-3,
    height_offset_max_m=1e-3,
    height_samples=2001,
    linearization_eps=1e-6,
    balance_cost=1000.0,
    balance_joint_cost=3.0,
    other_joint_cost=0.3,
    clip_controls=True,
    perturbations_enabled=True,
    perturb_seed=1,
    perturb_duration_s=6.0,
    perturb_ctrl_rate_s=0.8,
    perturb_balance_std=0.01,
    perturb_other_std=0.08,
)

CONFIG = SimpleNamespace(run=RUN_SETTINGS, controller=CONTROLLER)

__all__ = ["CONFIG", "RUN_SETTINGS", "CONTROLLER"]
