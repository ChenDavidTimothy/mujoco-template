import sys

import numpy as np

import mujoco_template as mt
from pendulum_common import (
    initialize_state as seed_pendulum,
    make_env as make_pendulum_env,
    make_tip_probes,
    resolve_pendulum_columns,
)
from pendulum_passive_config import CONFIG


def build_env():
    obs_spec = mt.ObservationSpec(
        include_ctrl=False,
        include_sensordata=False,
        include_time=True,
        sites_pos=("tip",),
    )
    return make_pendulum_env(obs_spec=obs_spec)


def seed_env(env):
    init_cfg = CONFIG.initial_state
    seed_pendulum(env, angle_deg=init_cfg.angle_deg, velocity_deg=init_cfg.velocity_deg)


def summarize(result):
    recorder = result.recorder
    rows = recorder.rows
    if not rows:
        print(f"Viewer closed. Final simulated time: {result.env.data.time:.3f}s")
        return

    columns = resolve_pendulum_columns(result.env.model)
    column_index = recorder.column_index
    stride = max(1, result.settings.simulation.sample_stride)

    time_idx = column_index[columns["time"]]
    angle_idx = column_index[columns["angle"]]
    velocity_idx = column_index[columns["velocity"]]
    tip_z_idx = column_index[columns["tip_z"]]

    for idx in range(0, len(rows), stride):
        row = rows[idx]
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


def main(argv=None):
    init_cfg = CONFIG.initial_state
    print(
        "Initial pendulum angle: {:.2f} deg; velocity: {:.2f} deg/s".format(
            init_cfg.angle_deg,
            init_cfg.velocity_deg,
        )
    )

    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
