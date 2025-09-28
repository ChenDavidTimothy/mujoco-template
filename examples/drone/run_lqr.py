from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))

try:  # Support both `python -m` and direct script execution.
    from .drone_config import CONFIG
    from .scenarios import HARNESS, summarize
except ImportError:  # pragma: no cover - fallback for `python examples/drone/run_lqr.py`
    CONFIG = import_module("examples.drone.drone_config").CONFIG  # type: ignore[attr-defined]
    _scenarios = import_module("examples.drone.scenarios")
    HARNESS = _scenarios.HARNESS
    summarize = _scenarios.summarize


def main(argv: list[str] | None = None) -> None:
    ctrl_cfg = CONFIG.controller
    traj_cfg = CONFIG.trajectory
    run_settings = CONFIG.run

    print(
        "Preparing drone LQR controller (keyframe {} | target [{:.2f}, {:.2f}, {:.2f}] m)".format(
            ctrl_cfg.keyframe,
            *np.asarray(ctrl_cfg.goal_position_m, dtype=float),
        )
    )
    print(
        "Start position [{:.2f}, {:.2f}, {:.2f}] m | duration {} steps".format(
            *np.asarray(traj_cfg.start_position_m, dtype=float),
            run_settings.simulation.max_steps,
        )
    )
    print(
        "Start orientation [{:.3f}, {:.3f}, {:.3f}, {:.3f}] wxyz".format(
            *np.asarray(traj_cfg.start_orientation_wxyz, dtype=float)
        )
    )
    print(
        "Start velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(traj_cfg.start_velocity_mps, dtype=float),
            *np.asarray(traj_cfg.start_angular_velocity_radps, dtype=float),
        )
    )
    print(
        "Goal velocity [{:.2f}, {:.2f}, {:.2f}] m/s | angular [{:.2f}, {:.2f}, {:.2f}] rad/s".format(
            *np.asarray(ctrl_cfg.goal_velocity_mps, dtype=float),
            *np.asarray(ctrl_cfg.goal_angular_velocity_radps, dtype=float),
        )
    )

    result = HARNESS.run_from_cli(CONFIG.run, args=argv)
    summarize(result)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main(sys.argv[1:])
    except KeyboardInterrupt:  # pragma: no cover - user interrupt
        sys.exit(130)

