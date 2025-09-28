from __future__ import annotations

import numpy as np

import mujoco_template as mt

from ..pendulum_common import make_env, make_tip_probes, resolve_pendulum_columns
from ..pendulum_common import initialize_state as seed_pendulum
from ..pendulum_passive_config import CONFIG, ExampleConfig


def build_env(config: ExampleConfig = CONFIG) -> mt.Env:
    del config  # Passive scenario uses the default configuration baked into the harness.
    obs_spec = mt.ObservationSpec(
        include_ctrl=False,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_env(obs_spec=obs_spec)


def seed_env(env: mt.Env, config: ExampleConfig = CONFIG) -> None:
    init_cfg = config.initial_state
    seed_pendulum(env, angle_deg=init_cfg.angle_deg, velocity_deg=init_cfg.velocity_deg)


def summarize(result: mt.PassiveRunResult) -> None:
    recorder = result.recorder
    rows = recorder.rows
    if not rows:
        print(f"Viewer closed. Final simulated time: {result.env.data.time:.3f}s")
        return

    columns = resolve_pendulum_columns(result.env.model)
    column_index = recorder.column_index
    time_idx = column_index[columns["time"]]
    angle_idx = column_index[columns["angle"]]
    velocity_idx = column_index[columns["velocity"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for row in rows:
        print(
            f"t={row[time_idx]:5.3f}s angle={np.rad2deg(row[angle_idx]):6.2f}deg "
            f"vel={np.rad2deg(row[velocity_idx]):6.2f}deg/s tip_z={row[tip_z_idx]:6.3f}m"
        )

    times = np.array([row[time_idx] for row in rows], dtype=float)
    tip_z = np.array([row[tip_z_idx] for row in rows], dtype=float)
    print(f"Tip height range: {tip_z.min():.4f} m to {tip_z.max():.4f} m over {times[-1]:.3f}s")
    print(f"Executed {result.steps} steps; final simulated time: {result.env.data.time:.3f}s")


HARNESS = mt.PassiveRunHarness(
    build_env,
    description="Passive pendulum example (MuJoCo Template)",
    seed_fn=seed_env,
    probes=make_tip_probes,
    start_message="Running passive pendulum rollout...",
)


__all__ = ["HARNESS", "build_env", "seed_env", "summarize"]

